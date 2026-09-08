#!/usr/bin/env python3
"""Training-layer UI: generate data, fine-tune, evaluate.

Deliberately separate from app.py (the model/inference layer). This app never loads a
model itself — it shells out to mlx_lm.lora and scripts/eval.py — so it stays responsive
and a crashed run can never take the UI down with it.

    python scripts/train_ui.py          # http://localhost:7861
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from credit.metrics import CONFUSION_HEADERS  # noqa: E402
from credit.schema import RATINGS  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-8bit"

# mlx_lm.lora progress lines, e.g.
#   Iter 25: Train loss 1.234, Learning Rate 1.000e-05, It/sec 1.2, ...
#   Iter 100: Val loss 1.100, Val took 5.6s
_TRAIN_LOSS = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
_VAL_LOSS = re.compile(r"Iter (\d+): Val loss ([\d.]+)")

# The HF Xet CDN backend has failed on this machine mid-download; the classic HTTP path
# resumes reliably. Also keep Python unbuffered so the log streams live.
_CHILD_ENV = {**os.environ, "HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1"}

_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def _venv_bin(name: str) -> str:
    """Prefer this repo's venv so the UI can be launched from any interpreter."""
    candidate = REPO / ".venv" / "bin" / name
    return str(candidate) if candidate.exists() else name


def _empty_loss_frame() -> pd.DataFrame:
    return pd.DataFrame({"iter": [], "loss": [], "split": []})


def _stop_running() -> str:
    global _proc
    with _lock:
        if _proc is None or _proc.poll() is not None:
            return "Nothing running."
        _proc.send_signal(signal.SIGINT)
        return "Sent interrupt — the run will stop after the current step."


def _stream(cmd: list[str], cwd: Path):
    """Run a child process, yielding its output line by line."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            yield None, "A run is already in progress. Stop it first."
            return
        _proc = subprocess.Popen(
            cmd, cwd=cwd, env=_CHILD_ENV, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
    proc = _proc
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line, None
    proc.wait()
    yield None, f"__EXIT__{proc.returncode}"


# ---------------------------------------------------------------- data


def generate_data(n, seed, balance):
    cmd = [_venv_bin("python"), "scripts/generate_data.py", "--n", str(int(n)), "--seed", str(int(seed)), "--out", "data"]
    if not balance:
        cmd.append("--no-balance")
    log = f"$ {' '.join(cmd)}\n\n"
    yield log
    for line, status in _stream(cmd, REPO):
        if line:
            log += line
            yield log
        elif status and status.startswith("__EXIT__"):
            code = status.removeprefix("__EXIT__")
            log += f"\n[exit code {code}]\n"
            yield log
        elif status:
            yield log + status


# ---------------------------------------------------------------- training


def start_training(
    model, data_dir, adapter_path, iters, batch_size, num_layers,
    learning_rate, max_seq_length, fine_tune_type, mask_prompt, save_every, seed,
):
    cmd = [
        _venv_bin("mlx_lm.lora"),
        "--model", model,
        "--train",
        "--data", data_dir,
        "--adapter-path", adapter_path,
        "--iters", str(int(iters)),
        "--batch-size", str(int(batch_size)),
        "--num-layers", str(int(num_layers)),
        "--learning-rate", str(learning_rate),
        "--max-seq-length", str(int(max_seq_length)),
        "--fine-tune-type", fine_tune_type,
        "--save-every", str(int(save_every)),
        "--seed", str(int(seed)),
        "--steps-per-report", "10",
        "--steps-per-eval", str(max(50, int(iters) // 10)),
    ]
    if mask_prompt:
        cmd.append("--mask-prompt")

    log = f"$ {' '.join(cmd)}\n\n"
    frame = _empty_loss_frame()
    rows: list[dict] = []
    yield log, frame, "Starting…"

    for line, status in _stream(cmd, REPO):
        if line:
            log += line
            m = _TRAIN_LOSS.search(line)
            if m:
                rows.append({"iter": int(m.group(1)), "loss": float(m.group(2)), "split": "train"})
            m = _VAL_LOSS.search(line)
            if m:
                rows.append({"iter": int(m.group(1)), "loss": float(m.group(2)), "split": "validation"})
            if rows:
                frame = pd.DataFrame(rows)
            last = rows[-1] if rows else None
            note = f"iter {last['iter']} · {last['split']} loss {last['loss']:.4f}" if last else "running…"
            yield log, frame, note
        elif status and status.startswith("__EXIT__"):
            code = int(status.removeprefix("__EXIT__"))
            verdict = "Finished." if code == 0 else f"FAILED (exit {code}) — see log."
            log += f"\n[exit code {code}]\n"
            yield log, frame, verdict
        elif status:
            yield log, frame, status


# ---------------------------------------------------------------- evaluation


def run_eval(model, adapter_path, test_file, limit, use_adapter):
    out = REPO / "eval_results" / ("tuned.json" if use_adapter else "base.json")
    cmd = [
        _venv_bin("python"), "scripts/eval.py",
        "--model", model,
        "--data", test_file,
        "--out", str(out),
    ]
    if use_adapter:
        cmd += ["--adapter-path", adapter_path]
    if limit:
        cmd += ["--limit", str(int(limit))]

    log = f"$ {' '.join(cmd)}\n\n"
    empty_summary = pd.DataFrame({"metric": [], "value": []})
    empty_conf = pd.DataFrame(columns=CONFUSION_HEADERS)
    yield log, empty_summary, empty_conf

    for line, status in _stream(cmd, REPO):
        if line:
            log += line
            yield log, empty_summary, empty_conf
        elif status and status.startswith("__EXIT__"):
            log += f"\n[exit code {status.removeprefix('__EXIT__')}]\n"
            summary, conf = _load_metrics(out)
            yield log, summary, conf
        elif status:
            yield log, empty_summary, empty_conf


def _load_metrics(path: Path):
    if not path.exists():
        return pd.DataFrame({"metric": ["(no results)"], "value": [""]}), pd.DataFrame(columns=CONFUSION_HEADERS)
    m = json.loads(path.read_text())
    summary = pd.DataFrame(
        {
            "metric": [
                "examples", "unparseable", "exact accuracy", "within-1-notch",
                "mean notch error", "IG/HY accuracy", "macro F1",
            ],
            "value": [
                m["n"], m["unparseable"], f"{m['exact_accuracy']:.1%}",
                f"{m['within_one_notch']:.1%}", f"{m['mean_notch_error']:.3f}",
                f"{m['ig_hy_accuracy']:.1%}", f"{m['macro_f1']:.3f}",
            ],
        }
    )
    per_class = pd.DataFrame(
        [
            {"rating": r, "precision": round(c["precision"], 2),
             "recall": round(c["recall"], 2), "f1": round(c["f1"], 2), "support": c["support"]}
            for r, c in m["per_class"].items()
        ]
    )
    return summary, per_class


def compare_runs():
    rows = []
    for label, name in (("fine-tuned", "tuned.json"), ("base model", "base.json")):
        path = REPO / "eval_results" / name
        if path.exists():
            m = json.loads(path.read_text())
            rows.append({
                "run": label,
                "exact acc": f"{m['exact_accuracy']:.1%}",
                "within-1": f"{m['within_one_notch']:.1%}",
                "mean notch err": round(m["mean_notch_error"], 3),
                "IG/HY acc": f"{m['ig_hy_accuracy']:.1%}",
                "macro F1": round(m["macro_f1"], 3),
            })
    if not rows:
        return pd.DataFrame({"run": ["Run an evaluation first (both with and without adapters)."]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- UI


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Credit Rating — Training Layer") as demo:
        gr.Markdown(
            "# Credit Rating — Training Layer\n"
            "Generate data, fine-tune LoRA adapters, and evaluate. Runs everything as "
            "subprocesses; no model is loaded in this process. "
            "The chat/inference UI is a separate app (`app.py`)."
        )

        with gr.Tab("1 · Data"):
            gr.Markdown(
                "Labels come from a deterministic scorecard, so they stay internally "
                "consistent. Balancing evens out the rating buckets — without it the AAA "
                "and CCC tails are nearly empty."
            )
            with gr.Row():
                n_examples = gr.Number(3000, label="Examples", precision=0)
                data_seed = gr.Number(17, label="Seed", precision=0)
                balance = gr.Checkbox(True, label="Balance rating buckets")
            gen_btn = gr.Button("Generate dataset", variant="primary")
            gen_log = gr.Textbox(label="Output", lines=18, max_lines=18, interactive=False)
            gen_btn.click(generate_data, [n_examples, data_seed, balance], gen_log)

        with gr.Tab("2 · Train"):
            with gr.Row():
                with gr.Column(scale=1):
                    model = gr.Textbox(DEFAULT_MODEL, label="Base model")
                    data_dir = gr.Textbox("data", label="Data directory")
                    adapter_path = gr.Textbox("adapters", label="Adapter output path")
                    with gr.Row():
                        iters = gr.Number(1000, label="Iterations", precision=0)
                        batch_size = gr.Number(4, label="Batch size", precision=0)
                    with gr.Row():
                        num_layers = gr.Number(16, label="LoRA layers", precision=0)
                        max_seq_length = gr.Number(2048, label="Max seq length", precision=0)
                    learning_rate = gr.Number(1e-5, label="Learning rate")
                    fine_tune_type = gr.Dropdown(
                        ["lora", "dora", "full"], value="lora", label="Fine-tune type"
                    )
                    with gr.Row():
                        save_every = gr.Number(200, label="Checkpoint every", precision=0)
                        train_seed = gr.Number(17, label="Seed", precision=0)
                    mask_prompt = gr.Checkbox(
                        True,
                        label="Mask prompt (train on completion only — strongly recommended)",
                    )
                    with gr.Row():
                        train_btn = gr.Button("Start training", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                    status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    loss_plot = gr.LinePlot(
                        _empty_loss_frame(), x="iter", y="loss", color="split",
                        title="Loss", height=260,
                    )
                    train_log = gr.Textbox(
                        label="Training log", lines=22, max_lines=22, interactive=False,
                        autoscroll=True,
                    )
            train_btn.click(
                start_training,
                [model, data_dir, adapter_path, iters, batch_size, num_layers,
                 learning_rate, max_seq_length, fine_tune_type, mask_prompt,
                 save_every, train_seed],
                [train_log, loss_plot, status],
            )
            stop_btn.click(lambda: _stop_running(), None, status)

        with gr.Tab("3 · Evaluate"):
            gr.Markdown(
                "Score the held-out test split. **Run it twice** — once with adapters and "
                "once without — so you can tell whether fine-tuning beat plain prompting. "
                "Ratings are ordinal, so *mean notch error* is the metric that matters most."
            )
            with gr.Row():
                eval_model = gr.Textbox(DEFAULT_MODEL, label="Base model")
                eval_adapter = gr.Textbox("adapters", label="Adapter path")
            with gr.Row():
                test_file = gr.Textbox("data/test.jsonl", label="Test file")
                eval_limit = gr.Number(50, label="Limit (0 = all)", precision=0)
                use_adapter = gr.Checkbox(True, label="Use adapters (uncheck for baseline)")
            eval_btn = gr.Button("Run evaluation", variant="primary")
            with gr.Row():
                summary_table = gr.Dataframe(label="Metrics", interactive=False)
                per_class_table = gr.Dataframe(label="Per-class", interactive=False)
            eval_log = gr.Textbox(label="Output", lines=14, max_lines=14, interactive=False, autoscroll=True)
            eval_btn.click(
                run_eval,
                [eval_model, eval_adapter, test_file, eval_limit, use_adapter],
                [eval_log, summary_table, per_class_table],
            )

            gr.Markdown("### Fine-tuned vs base")
            compare_btn = gr.Button("Compare saved runs")
            compare_table = gr.Dataframe(interactive=False)
            compare_btn.click(compare_runs, None, compare_table)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
