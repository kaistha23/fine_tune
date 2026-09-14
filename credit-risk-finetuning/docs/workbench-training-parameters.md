# Native training & evaluation — parameter guide

Reference for every control in the workbench **Runs → Native training & evaluation** panel
(`http://127.0.0.1:8090`). For each one: what the options mean, the range the server
accepts, what the code does with it, and the theory behind it.

Values and rules here are taken from the code, not the form: the form's `min`/`max` are
hints, and the server (`workbench/server.py` `Config`, `JobRequest`, `spec_for`) is what
actually accepts or rejects a job.

**Base model facts used below** — `mlx-community/Qwen3.5-9B-4bit`: 32 transformer blocks,
hidden size 4096, a hybrid layout of three linear-attention blocks to every one
full-attention block, 4-bit weights. Training always uses LoRA with the prompt masked out of
the loss and gradient checkpointing on; these are fixed, not options.

---

## 1. Job setup

### Task

| Option | Meaning |
|---|---|
| **Credit analysis** (`credit_analysis`) | The model reads a factsheet and evidence and returns a structured credit assessment. |
| **Structured query plan** (`query_plan`) | The model turns a request into a `QueryPlan` that the guarded compiler turns into SQL. The model never writes SQL. |

Each task trains its **own adapter**. The dataset, the prompt/schema version and any
adapter you evaluate must belong to the same task, or the job is refused.

*Theory:* the two tasks have different output distributions — long reasoned JSON versus a
short, tightly constrained plan. One adapter per task avoids one task's gradients degrading
the other (negative transfer) and lets each be evaluated and rolled back independently.

### Job

| Option | What it does |
|---|---|
| **Train** | Fine-tunes a fresh LoRA adapter on the dataset's `train` split, validating on `validation`. Evaluation model must be *Base model*. |
| **Evaluate dataset** | Generates answers for the chosen splits with the base model or a trained adapter, and scores them. Nothing is trained. |
| **Feedback regression** | Runs the *development checks* created from reviewer feedback that have independent expected results. Splits are forced to `development`. |

*Theory:* training and evaluation are separate jobs so a score can always be traced to an
exact, frozen checkpoint (its SHA-256 is recorded). Feedback regression is a **development**
signal — the checks came from cases people already looked at, so passing them does not
measure held-out accuracy.

### Dataset

A registered dataset in the **credit-workbench-v2** format whose manifest `task` matches the
selected task. The manifest hash is re-checked when the job starts; if the files changed
after registration, the job is refused.

### Generation profile

Used by evaluation and regression jobs. Each case is generated **3 times** with seeds
42, 43, 44; the three answers are compared for consistency.

| Profile | temperature | top_p | top_k | max new tokens | Use it for |
|---|---|---|---|---|---|
| **Deterministic evaluation** | 0 | 0 | 0 | 1024 | Comparing checkpoints or prompts. Greedy decoding — the same input gives the same output, so a difference is caused by the model, not by chance. |
| **Serving consistency** | 0.1 | 0.9 | 20 | 1024 | Checking the model stays stable under light sampling similar to serving. Disagreement across the 3 runs is a warning sign. |

*Theory:*
- **Temperature** divides the logits before softmax. 0 means always take the most likely
  token (greedy); higher values flatten the distribution and add randomness.
- **top_p** (nucleus sampling) keeps only the smallest set of tokens whose probabilities sum
  to *p*; **top_k** keeps only the *k* most likely tokens. Both trim the unlikely tail.
- Thinking mode is disabled in both profiles (`enable_thinking: false`) because the output
  must be structured JSON.

Comparisons between two runs are only allowed when they used the same generation profile.

### Cached base

Lists locally cached snapshots of `Qwen3.5-9B-4bit` from `~/.cache/huggingface/hub`.
Nothing is downloaded by the dashboard. An adapter can only be evaluated on the **same
snapshot** it was trained on.

*Theory:* a LoRA adapter is a small correction learned against specific frozen weights.
Applied to a different snapshot or quantisation, the correction no longer lines up.

### Prompt / schema version

The system prompt and output JSON schema the model is trained or evaluated with. Changing
the active version never alters past runs — each run keeps the version it froze.

*Theory:* because the prompt is masked out of the loss, the adapter learns to produce the
answer *conditioned on that exact prompt shape*. Training under one version and serving
under another wastes the run, and two adapters trained under different versions are not
comparable.

---

## 2. Evaluation selection

### Evaluation model

| Option | Meaning |
|---|---|
| **Base model** | The frozen base with no adapter. Always evaluate this too — it is the baseline a fine-tune must beat. |
| *A completed training run* | That run's adapter, for the same task and base snapshot. |

For **Train** jobs this must be *Base model*: training always starts a fresh adapter.

### Checkpoint

| Option | File | Meaning |
|---|---|---|
| **Best validation** | `best_adapters.safetensors` | The adapter at the validation report with the lowest loss, provided it improved after at least one optimizer update. |
| **Final** | `adapters.safetensors` | The adapter at the end of training (or at early stop). |

If the chosen file does not exist — for example validation never improved — the job is
refused rather than silently falling back.

*Theory:* validation loss usually falls, then rises as the model starts to overfit the
training set. "Best" picks the point before that rise. "Final" is useful when the run was
short and never overfit, or to compare against best. Lower loss is a training diagnostic,
not proof of better credit answers — judge with the evaluation scores.

### Evaluation splits

| Option | Meaning |
|---|---|
| **Validation + test + OOT** | All held-out data. Default. |
| **Validation only** | The split used during training for checkpoint selection. Slightly optimistic, because it influenced which checkpoint was chosen. |
| **Test only** | Held-out cases from the same time period, never seen in training or selection. |
| **OOT only** | *Out-of-time*: cases dated on or after the dataset's cutoff. |

*Theory:* credit data drifts — rates, policy and borrower behaviour change. A model that
does well on test but badly on OOT has learned patterns of the past that do not hold now.
OOT is the closest offline proxy for production performance. Scorecards report bootstrap
95% intervals, and slices below 10 cases are flagged as too small to conclude anything.

---

## 3. Training schedule

### Epochs — range **1 to 10**, default **2**

One epoch is one full pass over the training split.

*Theory:* too few epochs underfits; too many memorises the training examples, which on a
small dataset shows up quickly as rising validation loss. With hundreds to low thousands of
examples, 1–3 epochs is the usual range for LoRA.

### Sequence limit — range **256 to 8192** tokens, default **2304**

The maximum length of one full example (system prompt + user turn + target answer).

- **Training:** any example longer than this makes preflight **fail** and names the case.
  Examples are never silently truncated, because truncation would cut off the end of the
  JSON answer — the part the model is trained on.
- **Evaluation:** the context limit is `max(4096, sequence limit)`; a case whose prompt plus
  1024 answer tokens does not fit is refused.

*Theory:* attention memory grows with sequence length. Set this just above the longest
example preflight reports (`max_tokens`), not far above it.

### Learning rate — range **greater than 0, up to 0.001**, default **0.00002** (2e-5)

The step size of each weight update.

*Theory:* too high and loss spikes or diverges; too low and the adapter barely changes in
the available updates. LoRA commonly uses 1e-5 to 2e-4. Because this setup has few
optimizer updates (see the worked example below), very small rates may leave the adapter
nearly unchanged.

### Batch size — options **1** or **2**, default **1**

Examples processed together in one forward/backward pass (one *micro-batch*).

*Theory:* larger batches give smoother gradients but use more memory. On a 9B model in
unified memory, 1–2 is the practical limit; accumulation provides the larger effective batch.

### Accumulation — form allows **1 to 32**, default **8**

Number of micro-batches whose gradients are summed before one optimizer update.

**Effective batch = batch size × accumulation**, capped at **32** by the server.

> **Only 1, 2, 4 or 8 pass preflight from the dashboard.** The server requires the report,
> validation and save intervals to be exact multiples of accumulation, and the dashboard
> sends fixed intervals of 8, 80 and 80 micro-batches. 16 or 32 (or any value not dividing
> 8) are rejected with *"Report, validation and save intervals must align with accumulation"*.

*Theory:* accumulation simulates a bigger batch without the memory cost — the gradient is
averaged over more examples before the weights move, which makes updates less noisy.

### How epochs, batch and accumulation combine

```
micro-batches     = ceil( ceil(train_examples × epochs / batch_size) / accumulation ) × accumulation
optimizer updates = micro-batches / accumulation
```

| Train examples | Epochs | Batch | Accum | Micro-batches | Optimizer updates |
|---|---|---|---|---|---|
| 60 | 2 | 1 | 8 | 120 | **15** |
| 1,200 | 2 | 1 | 8 | 2,400 | **300** |
| 1,200 | 2 | 2 | 8 | 1,200 | **150** |

The weights change once per **optimizer update**, not per example. 15 updates is a
mechanics test; it will not move a 9B adapter meaningfully.

---

## 4. LoRA adapter

LoRA leaves the base weights frozen and learns a low-rank correction for selected linear
layers:

```
W' = W + scale × (B · A)        A: r × d_in,   B: d_out × r,   B starts at zero
```

Only `A` and `B` are trained. Because `B` starts at zero, the adapter begins as an exact
no-op and learns only what the data pushes it towards.

### LoRA rank — options **8**, **16**, **32**, default **16**

`r`, the width of the low-rank bottleneck.

*Theory:* higher rank can express more complex changes but trains more parameters and
overfits small datasets sooner. Parameters per adapted matrix are `r × (d_in + d_out)` —
for a 4096 × 4096 projection, rank 16 trains about 131 thousand parameters instead of
16.8 million. 8 suits small or narrow datasets; 32 suits larger, more varied ones.

### LoRA scale — options **1** or **2**, default **2**

Multiplier on the learned correction. In MLX this is the multiplier itself — equivalent to
`alpha / rank` in other libraries, so scale 2 with rank 16 matches the common alpha 32.

*Theory:* scale sets how strongly the adapter influences the output. It interacts with
learning rate — doubling scale is similar to a larger effective step for the adapter.
Change one at a time.

### Dropout — options **0** or **0.05**, default **0.05**

Probability of zeroing the adapter's input activations during training.

*Theory:* dropout is regularisation — it stops the adapter relying on any single feature and
reduces overfitting. 0 for very short runs or large datasets; 0.05 is a safe default for
small datasets.

### Adapted layers — options **16** or **32**, default **16**

How many transformer blocks receive adapters, counted **from the top** (the last blocks).
The base has 32, so 16 adapts the upper half and 32 adapts every block.

*Theory:* upper layers carry task- and format-specific behaviour; lower layers carry general
language features. Adapting the upper half is usually enough to teach an output format and
domain style, with half the adapter size and memory. Use 32 when the task needs deeper
changes and there is enough data.

### Target modules

Which linear layers inside each adapted block get LoRA.

| Option | Modules | Meaning |
|---|---|---|
| **All eligible linear** | every linear layer the trainer supports | Maximum capacity. Default. |
| **Attention only** | `q/k/v/o_proj` in full-attention blocks; `in_proj_*`, `out_proj` in linear-attention blocks | Changes *what the model attends to*. Smallest adapter. |
| **Attention + MLP** | the above plus `gate_proj`, `up_proj`, `down_proj` | Also changes the feed-forward layers, where much factual and domain knowledge is stored. |

*Theory:* attention decides which parts of the context matter; the MLP transforms each
position and holds much of the stored knowledge. Attention-only is often enough for format
and grounding behaviour; adding MLP helps domain vocabulary and reasoning at the cost of
more parameters. The resolved module list is recorded in the run for reproducibility.

---

## 5. Optimizer and learning-rate schedule

### Optimizer — options **Adam** or **AdamW**, default **Adam**

| Option | Meaning |
|---|---|
| **Adam** | Adaptive per-parameter step sizes from running averages of the gradient and its square. No weight decay. |
| **AdamW** | Adam with weight decay applied separately from the gradient update ("decoupled"). |

### Weight decay — range **0 to 0.1**, default **0**

Shrinks weights slightly towards zero every update. **Only allowed with AdamW** — the server
rejects a non-zero value with Adam.

*Theory:* another regulariser against overfitting. For LoRA it pulls the correction towards
zero, i.e. back towards the base model's behaviour. 0–0.01 is typical.

### Schedule — options **Constant** or **Cosine decay**, default **Constant**

| Option | Behaviour |
|---|---|
| **Constant** | The learning rate stays fixed for the whole run. |
| **Cosine decay** | Optional linear warm-up, then the rate falls along a half cosine from the learning rate to its minimum. |

Cosine decay, counted in **optimizer updates**:

```
warm-up  : rises linearly from  lr × min_ratio  to  lr   over  round(updates × warm-up ratio) updates
decay    : lr_t = min_lr + ½ (lr − min_lr)(1 + cos(π t / T))      min_lr = lr × min_ratio
```

*Theory:* large early steps on randomly-initialised adapter weights can destabilise
training; warm-up avoids that. Decaying towards the end lets the model settle into a minimum
instead of bouncing around it, which usually gives a better final checkpoint.

### Warm-up ratio — range **0 to 0.2**, default **0**

Fraction of optimizer updates spent warming up. **Requires Cosine decay** — the server
rejects warm-up with a constant schedule. Capped so at least one update remains for decay.

### Minimum LR ratio — range **0 to 1**, default **0.1**

The floor of the cosine schedule as a fraction of the learning rate (also the warm-up start).
0.1 means the rate ends at 10% of its peak. Ignored with a constant schedule.

---

## 6. Reproducibility and stopping

### Seed — any integer, default **42**

Fixes random initialisation of the adapter's `A` matrices, data shuffling and dropout masks.

*Theory:* two runs with the same seed, data and settings should match; changing only the
seed shows how much of a result is luck. On a small dataset, seed-to-seed variation can be
larger than the effect of a hyperparameter change — compare across a few seeds before
concluding.

### Early-stop patience — range **1 to 10**, default **3**

Number of consecutive validation reports without enough improvement before training stops.
Validation runs every **80 micro-batches** (10 updates at accumulation 8).

*Theory:* stops training once validation loss has stopped improving, saving time and
limiting overfitting. With patience 3, training continues through 240 micro-batches of no
improvement before stopping. On a short run with few validation reports, early stopping may
never trigger.

### Minimum loss improvement — range **0 to 1**, default **0.001**

How much validation loss must fall to count as an improvement, and so to update the best
checkpoint and reset patience.

*Theory:* filters out noise. Too large and genuine small gains are ignored; 0 lets random
fluctuations count as progress.

---

## 7. Fixed settings (not shown in the panel)

| Setting | Value | Why |
|---|---|---|
| Fine-tune type | LoRA | Base weights stay frozen and 4-bit. |
| Prompt masking | on | Loss is computed only on the answer, not on the factsheet and evidence. |
| Gradient checkpointing | on | Recomputes activations to fit a 9B model in unified memory. |
| Report interval | every 8 micro-batches | Training loss logged. |
| Validation interval | every 80 micro-batches | Drives best checkpoint and early stopping. |
| Save interval | every 80 micro-batches | Intermediate adapter saved. |
| Validation batches | all (`-1`) | Validation loss uses the whole validation split. |
| Effective batch cap | 32 | Local memory profile. |

---

## 8. Rules the server enforces

A job is refused if:

- report, validation or save interval is not a multiple of accumulation;
- weight decay is non-zero with Adam;
- warm-up is non-zero with a constant schedule;
- batch size × accumulation exceeds 32;
- the train or validation split is empty;
- any training example exceeds the sequence limit;
- the dataset, prompt/schema version or adapter belongs to a different task;
- the adapter was trained on a different base snapshot;
- the chosen checkpoint file does not exist;
- a Train job selects an existing adapter instead of Base model.

---

## 9. Suggested starting points

| Situation | Suggested settings |
|---|---|
| **Mechanics check** (does the pipeline run end to end) | defaults; epochs 1; small dataset |
| **Small dataset** (a few hundred examples) | rank 8, dropout 0.05, 16 layers, attention only or all linear, epochs 2–3, LR 1e-4, cosine with warm-up 0.1 |
| **Larger, varied dataset** (low thousands) | rank 16, all linear, 16 layers, epochs 2, LR 5e-5 to 1e-4, cosine with warm-up 0.05, AdamW with weight decay 0.01 |
| **Comparing two runs** | Deterministic evaluation profile, Validation + test + OOT, same seed and same prompt/schema version |

Change one setting at a time, always evaluate the base model alongside the adapter, and
treat validation loss as a diagnostic — the evaluation scores on test and OOT decide.
