#!/usr/bin/env python3
"""Gradio chat UI for the fine-tuned credit rating model.

Works against any OpenAI-compatible MLX server. Two options:

  oMLX (default here, port 8000) — needs a FUSED model, it does not load adapters:
      mlx_lm.fuse --model mlx-community/Qwen2.5-7B-Instruct-8bit \
                  --adapter-path ./adapters \
                  --save-path ~/models/credit-rating-qwen2.5-7b
      omlx serve --model-dir ~/models
      python app.py

  mlx_lm.server (port 8080) — loads adapters directly, no fusing needed:
      mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-8bit \
                    --adapter-path ./adapters --port 8080
      python app.py --base-url http://localhost:8080/v1

oMLX also ships its own chat UI at http://localhost:8000/admin/chat; this app adds the
structured issuer form that formats prompts exactly as the model was trained on.
"""

from __future__ import annotations

import argparse

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from openai import OpenAI

from credit.schema import (
    SYSTEM_PROMPT, Issuer, format_profile, parse_rating, rating_index,
)
from credit.scorecard import INDUSTRIES, RATE_ENVIRONMENTS, rate_issuer, sample_issuer

# This machine's oMLX is configured for port 9905 (see ~/.omlx/settings.json → server.port);
# oMLX's own default is 8000 and mlx_lm.server uses 8080. Override with --base-url.
DEFAULT_BASE_URL = "http://localhost:9905/v1"
SECTORS = list(INDUSTRIES)

_client: OpenAI | None = None
_model_name = "default_model"


def _stream(messages: list[dict], max_tokens: int, temperature: float):
    """Yield the accumulating assistant response."""
    assert _client is not None
    try:
        stream = _client.chat.completions.create(
            model=_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
    except Exception as exc:
        yield (
            f"**Could not reach the model server.**\n\n`{exc}`\n\n"
            "Start one of these, then retry:\n\n"
            "```\nomlx serve --model-dir ~/models          # port 8000, needs a fused model\n"
            "mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-8bit \\\n"
            "              --adapter-path ./adapters --port 8080\n```"
        )
        return

    accumulated = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            accumulated += delta
            yield accumulated


def rate_issuer_ui(
    name, sector, revenue, growth, margin, leverage, coverage,
    current, cyclicality, position, rate_env, max_tokens, temperature,
):
    issuer = Issuer(
        name=name or "Unnamed Issuer",
        sector=sector,
        revenue_musd=float(revenue),
        revenue_growth=float(growth) / 100.0,
        ebitda_margin=float(margin) / 100.0,
        debt_to_ebitda=float(leverage),
        interest_coverage=float(coverage),
        current_ratio=float(current),
        cyclicality=int(cyclicality),
        competitive_position=int(position),
        rate_environment=rate_env,
    )
    # Formatting through the shared helper guarantees the UI sends the exact layout the
    # model was fine-tuned on.
    profile = format_profile(issuer)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": profile},
    ]
    for partial in _stream(messages, max_tokens, temperature):
        yield partial, profile


def _complete(messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Non-streaming completion — scoring needs the whole answer before it can parse."""
    assert _client is not None
    resp = _client.chat.completions.create(
        model=_model_name, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _score(issuer_or_profile, truth: str, response: str, source: str, history: list) -> tuple:
    """Append one scored answer to the session history and recompute the tally."""
    predicted = parse_rating(response)
    if predicted is None:
        notch, ok = None, "unparseable"
    else:
        notch = abs(rating_index(truth) - rating_index(predicted))
        ok = "correct" if notch == 0 else ("1 notch" if notch == 1 else f"{notch} notches")

    history = history + [{
        "#": len(history) + 1,
        "source": source,
        "model": predicted or "—",
        "truth": truth,
        "notch err": "—" if notch is None else notch,
        "result": ok,
    }]

    scored = [h for h in history if h["notch err"] != "—"]
    if scored:
        exact = sum(1 for h in scored if h["notch err"] == 0)
        mean_notch = sum(h["notch err"] for h in scored) / len(scored)
        tally = (
            f"**Session: {exact}/{len(history)} exact** · "
            f"mean notch error {mean_notch:.2f} · "
            f"within 1 notch {sum(1 for h in scored if h['notch err'] <= 1)}/{len(history)}"
        )
    else:
        tally = "**Session: no parseable answers yet**"

    verdict = (
        f"### Model: `{predicted or '—'}`  ·  Truth: `{truth}`  ·  {ok}\n\n{response}"
    )
    return verdict, pd.DataFrame(history), tally, history


def ask_and_score(
    name, sector, revenue, growth, margin, leverage, coverage,
    current, cyclicality, position, rate_env, max_tokens, history,
):
    """Score a profile you typed. Ground truth is free — we own the scorecard."""
    issuer = Issuer(
        name=name or "Unnamed Issuer", sector=sector, revenue_musd=float(revenue),
        revenue_growth=float(growth) / 100.0, ebitda_margin=float(margin) / 100.0,
        debt_to_ebitda=float(leverage), interest_coverage=float(coverage),
        current_ratio=float(current), cyclicality=int(cyclicality),
        competitive_position=int(position), rate_environment=rate_env,
    )
    # rng=None -> no analyst-judgment noise, so this is the scorecard's exact answer.
    truth, _ = rate_issuer(issuer, rng=None)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_profile(issuer)},
    ]
    try:
        response = _complete(messages, max_tokens, 0.0)
    except Exception as exc:
        return f"**Could not reach the model server.**\n\n`{exc}`", pd.DataFrame(history), "", history
    return _score(issuer, truth, response, "typed", history)


def randomize_issuer():
    """Fill the form with a fresh sampled issuer spanning the full stress range."""
    rng = np.random.default_rng()
    issuer = sample_issuer(rng, stress=float(rng.uniform(-1.2, 1.2)))
    return (
        issuer.name, issuer.sector, round(issuer.revenue_musd),
        round(issuer.revenue_growth * 100, 1), round(issuer.ebitda_margin * 100, 1),
        round(issuer.debt_to_ebitda, 2), round(issuer.interest_coverage, 2),
        round(issuer.current_ratio, 2), issuer.cyclicality,
        issuer.competitive_position, issuer.rate_environment,
    )


def draw_test_example(test_path: str):
    """Pull a random held-out example; its stored label is the ground truth."""
    path = Path(test_path)
    if not path.exists():
        return f"No test file at {test_path}", ""
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not records:
        return "Test file is empty.", ""
    record = records[np.random.default_rng().integers(len(records))]["messages"]
    profile = next(m["content"] for m in record if m["role"] == "user")
    truth = parse_rating(next(m["content"] for m in record if m["role"] == "assistant"))
    return profile, truth


def score_test_example(profile, truth, max_tokens, history):
    if not profile or not truth:
        return "Draw an example first.", pd.DataFrame(history), "", history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": profile},
    ]
    try:
        response = _complete(messages, max_tokens, 0.0)
    except Exception as exc:
        return f"**Could not reach the model server.**\n\n`{exc}`", pd.DataFrame(history), "", history
    return _score(profile, truth, response, "test set", history)


def chat_fn(message, history, max_tokens, temperature):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    yield from _stream(messages, max_tokens, temperature)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Corporate Credit Rating Model") as demo:
        gr.Markdown(
            "# Corporate Credit Rating Model\n"
            "Fine-tuned on MLX. Ratings are on the AAA / AA / A / BBB / BB / B / CCC scale."
        )
        with gr.Accordion("Generation settings", open=False):
            max_tokens = gr.Slider(32, 512, value=200, step=8, label="Max tokens")
            temperature = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05, label="Temperature (0 = deterministic)"
            )

        with gr.Tab("Rate an issuer"):
            with gr.Row():
                with gr.Column(scale=1):
                    name = gr.Textbox("Northwind Power Corp", label="Company")
                    sector = gr.Dropdown(SECTORS, value="Utilities", label="Sector")
                    revenue = gr.Number(5000, label="Revenue ($M)")
                    growth = gr.Number(1.8, label="Revenue growth (%)")
                    margin = gr.Number(31.0, label="EBITDA margin (%)")
                    leverage = gr.Number(4.20, label="Total debt / EBITDA (x)")
                    coverage = gr.Number(2.10, label="EBIT / interest expense (x)")
                    current = gr.Number(1.05, label="Current ratio")
                    cyclicality = gr.Slider(1, 5, value=1, step=1, label="Industry cyclicality")
                    position = gr.Slider(1, 5, value=4, step=1, label="Competitive position")
                    rate_env = gr.Dropdown(
                        RATE_ENVIRONMENTS, value="Rising", label="Rate environment"
                    )
                    submit = gr.Button("Assess rating", variant="primary")
                with gr.Column(scale=1):
                    output = gr.Markdown(label="Assessment")
                    sent_prompt = gr.Textbox(
                        label="Prompt sent to the model", lines=14, interactive=False
                    )

            # Cyclicality is a property of the sector in the training data, so track it.
            sector.change(
                lambda s: gr.update(value=INDUSTRIES[s]["cyclicality"]),
                inputs=sector,
                outputs=cyclicality,
            )
            submit.click(
                rate_issuer_ui,
                inputs=[name, sector, revenue, growth, margin, leverage, coverage,
                        current, cyclicality, position, rate_env, max_tokens, temperature],
                outputs=[output, sent_prompt],
            )

        with gr.Tab("QA / Accuracy"):
            gr.Markdown(
                "Every answer is scored automatically. For a profile you type, the "
                "deterministic scorecard supplies ground truth — so you can invent any "
                "issuer and still get an objective result. History is session-only."
            )
            history_state = gr.State([])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Ask about a profile you choose")
                    qa_name = gr.Textbox("Ashford Retail Group", label="Company")
                    qa_sector = gr.Dropdown(SECTORS, value="Retail", label="Sector")
                    with gr.Row():
                        qa_revenue = gr.Number(1800, label="Revenue ($M)")
                        qa_growth = gr.Number(-2.0, label="Growth (%)")
                    with gr.Row():
                        qa_margin = gr.Number(7.5, label="EBITDA margin (%)")
                        qa_leverage = gr.Number(6.10, label="Debt/EBITDA (x)")
                    with gr.Row():
                        qa_coverage = gr.Number(1.40, label="EBIT/interest (x)")
                        qa_current = gr.Number(0.95, label="Current ratio")
                    with gr.Row():
                        qa_cyclicality = gr.Slider(1, 5, value=4, step=1, label="Cyclicality")
                        qa_position = gr.Slider(1, 5, value=2, step=1, label="Position")
                    qa_rate_env = gr.Dropdown(
                        RATE_ENVIRONMENTS, value="Rising", label="Rate environment"
                    )
                    with gr.Row():
                        qa_ask = gr.Button("Ask & score", variant="primary")
                        qa_random = gr.Button("Randomize")

                    gr.Markdown("#### …or draw a held-out test example")
                    qa_test_path = gr.Textbox("data/test.jsonl", label="Test file")
                    with gr.Row():
                        qa_draw = gr.Button("Draw example")
                        qa_score_drawn = gr.Button("Score drawn example", variant="primary")
                    qa_drawn = gr.Textbox(label="Drawn profile", lines=6, interactive=False)
                    qa_truth = gr.Textbox(label="Its true rating", interactive=False)

                with gr.Column(scale=1):
                    qa_verdict = gr.Markdown("Ask something to begin.")
                    qa_tally = gr.Markdown()
                    qa_history = gr.Dataframe(label="Session history", interactive=False)
                    qa_clear = gr.Button("Clear history")

            qa_max_tokens = gr.Slider(
                32, 512, value=200, step=8, label="Max tokens", visible=False
            )
            qa_form = [qa_name, qa_sector, qa_revenue, qa_growth, qa_margin, qa_leverage,
                       qa_coverage, qa_current, qa_cyclicality, qa_position, qa_rate_env]

            qa_ask.click(
                ask_and_score,
                inputs=[*qa_form, qa_max_tokens, history_state],
                outputs=[qa_verdict, qa_history, qa_tally, history_state],
            )
            qa_random.click(randomize_issuer, None, qa_form)
            qa_draw.click(draw_test_example, qa_test_path, [qa_drawn, qa_truth])
            qa_score_drawn.click(
                score_test_example,
                inputs=[qa_drawn, qa_truth, qa_max_tokens, history_state],
                outputs=[qa_verdict, qa_history, qa_tally, history_state],
            )
            qa_clear.click(
                lambda: ("Cleared.", pd.DataFrame(), "", []),
                None,
                [qa_verdict, qa_history, qa_tally, history_state],
            )

        with gr.Tab("Free chat"):
            gr.ChatInterface(
                chat_fn,
                additional_inputs=[max_tokens, temperature],
                # Gradio requires list-of-lists once additional_inputs are declared:
                # each row is [message, max_tokens, temperature].
                examples=[
                    ["What does a 6.0x debt/EBITDA ratio imply for a retailer's rating?", 200, 0.0],
                    ["Why do utilities sustain higher leverage at the same rating?", 200, 0.0],
                ],
            )
    return demo


def _resolve_model(client: OpenAI, requested: str | None) -> str:
    """Ask the server what it is serving.

    oMLX identifies models by directory name or alias while mlx_lm.server calls its
    single model "default_model", so hardcoding either one breaks against the other.
    """
    if requested:
        return requested
    try:
        available = [m.id for m in client.models.list().data]
    except Exception as exc:
        print(f"Could not list models ({exc}); falling back to 'default_model'.")
        return "default_model"
    if not available:
        return "default_model"
    if len(available) > 1:
        print(f"Server offers {len(available)} models: {', '.join(available)}")
        print(f"Using {available[0]!r} — override with --model.")
    return available[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--model", default=None, help="Model id. Auto-detected from /v1/models if omitted."
    )
    parser.add_argument("--api-key", default="not-needed", help="Needed if oMLX ran with --api-key.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    global _client, _model_name
    _client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    _model_name = _resolve_model(_client, args.model)

    print(f"Model server: {args.base_url}")
    print(f"Model:        {_model_name}")
    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    build_ui().launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
