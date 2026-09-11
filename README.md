# fine_tune

This repository holds **two independent projects**. They share no code, no data and no
dependencies; pick the one you want.

| Directory | Project | Use it for |
|---|---|---|
| `.` (this README, below) | **Corporate credit rating** - LoRA fine-tune on MLX | A single model that assigns a rating (AAA-CCC) with a rationale. Self-contained, `pip`/`requirements.txt`, Qwen2.5-7B |
| [`credit-risk-finetuning/`](credit-risk-finetuning/) | **Credit-risk advisory copilot** - governed pipeline | A guarded system: default-deny SQL, point-in-time correctness, ACL-aware RAG across SAMA/CBUAE, human SQL review, action control, release gates, and a correction loop that feeds fine-tuning. `uv`, Qwen3 series |

The second is the larger of the two: a Docker-plus-native-Mac architecture where the API
never touches the database, only a restricted data service reads curated data (read-only),
and every answer is checked against its evidence before release. Start at
[`credit-risk-finetuning/README.md`](credit-risk-finetuning/README.md) — it carries its own
quick start, architecture notes and a **What is left** section.

Everything below this line documents the first project only.

---

# Corporate credit rating — LoRA fine-tune on MLX

Fine-tunes an open LLM on Apple Silicon to assign a **corporate credit rating** (AAA through
CCC) with a short rationale, from an issuer's financial and business profile.

This is a **rating classification** model, not a PD (probability-of-default) model. It predicts
an ordinal rating category and explains it in prose; it does not estimate a default probability.

Base model: `mlx-community/Qwen2.5-7B-Instruct-8bit` (~8GB, no license gate).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python scripts/generate_data.py --n 300000 --out ./data

# 2. Fine-tune LoRA adapters (~20-40 min on an M-series Pro)
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-8bit \
  --train --data ./data \
  --iters 1000 --batch-size 4 --num-layers 16 \
  --learning-rate 1e-5 --mask-prompt \
  --adapter-path ./adapters

# 3. Evaluate — run twice to prove the fine-tune beat the base model
python scripts/eval.py --adapter-path ./adapters --out eval_results/tuned.json
python scripts/eval.py                            --out eval_results/base.json

# 4. Fuse the adapters into a standalone MLX model (required for oMLX).
#    ~/.omlx/models is oMLX's default model dir, so serving needs no extra flag.
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-7B-Instruct-8bit \
  --adapter-path ./adapters \
  --save-path ~/.omlx/models/credit-rating-qwen2.5-7b

# 5. Serve it and chat. This machine's oMLX is configured for port 9905
#    (~/.omlx/settings.json → server.port) and auto-starts with the menu bar app.
omlx serve                          # or just use the already-running app
python app.py                       # opens http://localhost:7860
```

oMLX also has its own chat UI at <http://localhost:9905/admin/chat>. `app.py` adds the
structured issuer form that builds prompts in the exact format the model was trained on.

---

## How the synthetic data works

The hard part of synthetic training data is **label consistency**. If you ask an LLM to invent
financials, a rating, and a rationale in one shot, the rating drifts — the same profile gets rated
differently across examples, and the model has no coherent mapping to learn. So generation is
split in two:

**Step A — the label comes from a deterministic scorecard** ([credit/scorecard.py](credit/scorecard.py)).
Eight weighted factors (leverage, interest coverage, EBITDA margin, scale, liquidity, growth,
sector cyclicality, competitive position) each score 0-100 by piecewise-linear interpolation, and
the weighted composite maps to a rating band. Leverage is divided by a **sector tolerance
multiplier** first, which is why a utility at 4.2x debt/EBITDA rates higher than a tech company at
the same leverage. A rising-rate environment applies a penalty that scales with the debt load.

Ten sector archetypes (`INDUSTRIES`) define the sampling distributions. A per-issuer `stress`
parameter in [-1.2, 1.2] slides the whole profile toward distress or strength — **without it,
sector distributions alone bunch everything into A through BB and the AAA and CCC tails end up
essentially empty.** Rejection sampling then evens out the buckets, and the train/valid/test split
is stratified so every split covers the full scale.

A small Gaussian noise term is added to the composite before banding. Without it the mapping is
perfectly separable and the model memorizes exact cut points instead of learning the ordering.

**Step B — the rationale is composed from the same factor scores**
([credit/rationale.py](credit/rationale.py)), *after* the rating is fixed. It names the highest-
salience strengths and weaknesses with their actual figures. Because it reads from the scorecard's
own output, a rationale can never argue for a different rating than its label.

The rationale is templated rather than LLM-written. That trades linguistic variety for guaranteed
consistency and no API dependency. To get richer prose, replace `compose_rationale` with a call to
an LLM that is **given** the decided rating and told to justify it — never to choose it.

### The caveat that matters

Training on synthetic labels teaches the model **this scorecard's logic**, not real agency
behavior. Its ceiling is the scorecard itself. Use it to prove out the pipeline and the I/O
format; if the goal is matching real S&P/Moody's/Fitch ratings, you need real rating histories
(licensed — WRDS / Compustat / Capital IQ; not freely scrapable). Company financials themselves
are free from SEC EDGAR/XBRL, so you can swap real inputs under synthetic labels as a middle step.

---

## Data format

`data/{train,valid,test}.jsonl`, one JSON object per line in mlx-lm chat format:

```json
{"messages": [
  {"role": "system",    "content": "You are a corporate credit rating analyst. ..."},
  {"role": "user",      "content": "Assess the credit rating for the following issuer.\n\nCompany: ...\nSector: Utilities\nTotal debt / EBITDA: 4.15x\n..."},
  {"role": "assistant", "content": "Rating: BB\n\nRationale: Elevated leverage of 4.2x debt/EBITDA and thin interest coverage of 2.1x constrain the credit profile. ..."}
]}
```

Prompts are built by `format_profile()` in [credit/schema.py](credit/schema.py), which the UI and
eval script also call — so the model never sees a layout it wasn't trained on.

### Why `--mask-prompt`

It trains loss on the **completion only**. Measured on this dataset (2,398 training records
tokenized through mlx-lm's own `ChatDataset`):

```
mean prompt masked:      168 tokens
mean completion trained:  66 tokens
--> only 28.4% of tokens are actually trained on
```

Without masking, **71.6% of the gradient signal** would go into reproducing the templated issuer
profile rather than learning the ratio-to-rating mapping. The mask lands exactly on the assistant
turn boundary — everything through `<|im_start|>assistant\n` is masked, and training starts at
`Rating: ...`.

Longest sequence is 266 tokens, well inside the 2048 default, so nothing is truncated.

---

## Evaluation

Ratings are **ordinal** — one notch off is a much smaller error than five. `scripts/eval.py`
reports:

| Metric | Why |
|---|---|
| **Mean notch error** | Primary metric. Distance on the rating scale, so near-misses are scored as near-misses. |
| Exact bucket accuracy | Strict match. |
| Within-1-notch accuracy | Forgiving progress tracker across runs. |
| **IG/HY crossover accuracy** | The BBB/BB boundary drives eligibility rules and funding costs — the economically meaningful line. |
| Macro F1 | Stops a model that always guesses the majority bucket from looking good. |
| Confusion matrix | Exposes systematic bias, e.g. rating consistently one notch generous. |
| Unparseable count | Responses with no extractable rating; counted as failures, not skipped. |

Decoding is greedy (`temp=0.0`) so results are reproducible.

**Always run the base-model baseline too.** If the fine-tune doesn't beat plain prompting, the
fine-tune isn't earning its keep.

---

## Serving: oMLX vs `mlx_lm.server`

Both expose an OpenAI-compatible API, so [app.py](app.py) works against either — only the
port and the model id differ, and the app auto-detects the model id from `/v1/models`.

| | oMLX | `mlx_lm.server` |
|---|---|---|
| Port | **9905 on this machine** (upstream default 8000) | 8080 |
| LoRA adapters | **Not supported — must fuse first** | Loads `--adapter-path` directly |
| Built-in chat UI | Yes, `/admin/chat` | No |
| Menu bar app | Yes (native Swift) | No |
| Anthropic `/v1/messages` | Yes | No |
| Continuous batching, SSD KV cache | Yes | No |
| Trains models | No — inference only | No (training is `mlx_lm.lora`) |

**Neither one trains.** oMLX is an inference server; the fine-tune always runs through
`mlx_lm.lora`. oMLX replaces step 5, not steps 1-3.

### The fusing requirement

oMLX discovers whole MLX model directories and has no adapter-loading flag, so LoRA adapters
must be baked in with `mlx_lm.fuse` first. `mlx_lm.server` can skip this and load `./adapters`
live, which is faster to iterate on — use it while tuning, and fuse for oMLX once you're happy.

### Model directory layout

`--model-dir` defaults to `~/.omlx/models`. Each subdirectory must contain `config.json` and
`*.safetensors` — exactly what `mlx_lm.fuse` writes. The directory name becomes the API model id.

```
~/.omlx/models/
├── credit-rating-qwen2.5-7b/     <- our fused model, served as "credit-rating-qwen2.5-7b"
└── some-other-model/
```

`app.py` reads the id back from `/v1/models`, so it works without being told the name.

### Useful oMLX flags

```bash
omlx serve \
  --memory-guard balanced \         # one of: safe | balanced | aggressive
  --memory-guard-gb 48 \            # cap usage; this Mac has 64GB
  --paged-ssd-cache-dir ~/.omlx/cache \
  --max-concurrent-requests 16 \
  --api-key your-secret-key         # then: python app.py --api-key your-secret-key
```

`omlx start` / `stop` / `restart` run it as a managed background service. `omlx launch <tool>`
wires it into external tools, and `omlx diagnose menubar` debugs the menu bar app.

### The existing setup on this machine

oMLX **0.5.3 was already installed** here on 2026-07-25 via the macOS app
(`/Applications/oMLX.app`), which symlinks `/opt/homebrew/bin/omlx → ~/.omlx/bin/omlx`. It is not
a Homebrew package — `brew list` doesn't know it, and the `jundot/omlx` tap is not present.

Existing state worth knowing before you add to it:

- **Port 9905**, not the documented 8000 (`~/.omlx/settings.json` → `server.port`).
- `auto_start_on_launch: true` — the server comes up with the menu bar app.
- Model dir is `~/.omlx/models`, already holding **`mlx-community/Qwen3.6-35B-A3B-6bit` (27GB)**
  in the two-level layout. Our ~8GB fused model sits alongside it; both fit in 64GB but only one
  loads at a time under the LRU memory manager.
- `memory_guard_tier: balanced`, SSD cache enabled.

Two documented constraints that turn out not to bite:

- **Python**: oMLX documents 3.11–3.13 and this Mac defaults to **3.14.6**. The app bundle ships
  its own runtime, so it runs fine. A `pip install -e .` source install would hit this —
  `/opt/homebrew/bin/python3.11` is present if you ever need that route.
- **Chip**: the docs list M1/M2/M3/M4 and this is an **M5 Pro**. It runs, but it is newer than
  anything the project explicitly claims support for.

## How the model is saved, and how to ship it

**Nothing is pickled.** MLX writes `.safetensors`, not `.pt`/`.bin`/`pickle`. That matters for
more than tidiness: unpickling *executes arbitrary Python*, so a downloaded pickle checkpoint is
remote code execution waiting to happen. A safetensors file is a JSON header plus raw tensor
bytes — pure data, memory-mappable, zero-copy, no code path on load. Treat any `.pt` weights you
receive from outside as untrusted; safetensors you can simply load.

### What training produces

`mlx_lm.lora` writes two files into `--adapter-path`:

```
adapters/
├── adapters.safetensors      # the trained LoRA weights (mx.save_safetensors)
└── adapter_config.json       # every training arg — rank, num_layers, lr, fine_tune_type
```

Plus `0000200_adapters.safetensors`-style checkpoints if you pass `--save-every`. Keep
`adapter_config.json` with the weights: it records the architecture the adapters assume, and
loading them against a different `--num-layers` will not line up.

The adapter is **only the delta** — a few MB, because LoRA trains small low-rank matrices on 16
layers rather than all 7B parameters. The base model is untouched.

### What fusing produces

`mlx_lm.fuse` merges the adapter into the weights and writes a complete, standalone MLX model —
roughly 8GB for this 8-bit 7B base:

```
credit-rating-qwen2.5-7b/
├── config.json
├── model-00001-of-*.safetensors    # merged weights
├── model.safetensors.index.json    # shard map
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
└── special_tokens_map.json, added_tokens.json
```

### Three ways to ship it

**1. Adapter only — a few MB.** Best when the recipient can pull the base model themselves.

```bash
tar czf credit-adapters.tar.gz adapters/
# on the other machine — base model downloads automatically:
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-8bit \
                --adapter-path ./adapters --prompt "..."
```

The base model id is the contract. Adapters are only valid against the exact base they were
trained on — same repo, same quantization.

**2. Fused model — ~8GB, self-contained.** Needed for oMLX, and for anywhere you don't want a
dependency on fetching the base.

```bash
mlx_lm.fuse --model mlx-community/Qwen2.5-7B-Instruct-8bit \
            --adapter-path ./adapters \
            --save-path ./credit-rating-qwen2.5-7b
tar czf credit-model.tar.gz credit-rating-qwen2.5-7b/
```

**3. Hugging Face Hub** — `fuse` uploads directly:

```bash
huggingface-cli login
mlx_lm.fuse --model mlx-community/Qwen2.5-7B-Instruct-8bit \
            --adapter-path ./adapters \
            --save-path ./credit-rating-qwen2.5-7b \
            --upload-repo your-username/credit-rating-qwen2.5-7b
```

Use a **private** repo. This model is trained on a synthetic scorecard, and a public rating model
invites being taken more seriously than its provenance supports. The uploader reads a `README.md`
model card from the save path — if the upload errors on a missing card, drop a `README.md` in
that directory and rerun.

### Which to choose

| Situation | Ship |
|---|---|
| Colleague already has the base model | Adapter (a few MB) |
| Serving via oMLX | Fused (no adapter support) |
| Air-gapped or reproducible deploy | Fused |
| Iterating on training | Adapter — `mlx_lm.server` hot-loads it, no fuse step |

## The two UIs

Training and inference are deliberately separate apps. The training layer never loads a model
into its own process — it shells out to `mlx_lm.lora` and `scripts/eval.py` — so the UI stays
responsive and a crashed run can't take it down.

```bash
python scripts/train_ui.py     # :7861  training layer  — no model server needed
python app.py                  # :7860  model/inference — needs oMLX running
```

**`scripts/train_ui.py` — training layer**

| Tab | What it does |
|---|---|
| 1 · Data | Regenerate the dataset (size, seed, bucket balancing) with live output |
| 2 · Train | Every LoRA parameter as a form, start/stop, streaming log, live loss curve |
| 3 · Evaluate | Run the evaluator, metrics + per-class tables, and a fine-tuned vs base comparison |

Training runs as a subprocess with `HF_HUB_DISABLE_XET=1` set, since the Xet CDN backend has
failed mid-download on this machine. Stop sends SIGINT so the run halts cleanly after the current
step. Train and validation loss are parsed out of the log and plotted as they arrive.

**`app.py` — model/inference layer**

| Tab | What it does |
|---|---|
| Rate an issuer | Structured form → streamed assessment, plus the exact prompt sent |
| QA / Accuracy | Every answer auto-scored; session history and running tally |
| Free chat | Open-ended conversation with the model |

### How QA auto-scoring works

Because the scorecard is ours, **ground truth is free for any profile you invent** — no labelling
step. Type an issuer, and `rate_issuer(..., rng=None)` computes the scorecard's exact answer
(noise off) to score the model against. "Randomize" samples a fresh issuer across the full stress
range so you can probe the tails, and "Draw example" pulls a held-out record from
`data/test.jsonl` and scores against its stored label.

The tally reports exact accuracy, mean notch error, and within-1-notch. Answers with no
extractable rating count against accuracy but are excluded from the notch average — they're
failures, not near-misses. History is session-only and clears on restart.

## Layout

```
credit/schema.py      Rating scale, Issuer, prompt/answer format, rating parsing
credit/scorecard.py   Sector archetypes, weighted scorecard, issuer sampling
credit/rationale.py   Rationale composition from factor scores
credit/metrics.py     Ordinal metrics — shared by the CLI evaluator and the training UI
scripts/generate_data.py
scripts/eval.py       CLI evaluator
scripts/train_ui.py   Training layer UI  (:7861)
app.py                Inference layer UI (:7860)
```

## Extending to the full 21-notch scale

`RATINGS` in [credit/schema.py](credit/schema.py) uses 7 major categories. The notched scale
(AA+, AA, AA-, ...) needs proportionally more data per class — plan on 10k+ examples — and finer
`_RATING_FLOORS` bands in the scorecard. Every metric in `eval.py` works unchanged, since they all
derive from index positions in `RATINGS`.

## Notes

- Adapters can be hot-swapped per request via an `"adapters"` field in the server's JSON body —
  no restart needed.
- `mlx_lm.fuse` produces a standalone model directory if you want to ship one artifact.
- Adapters and draft models are not supported in mlx-lm's distributed mode (irrelevant on a
  single Mac).
