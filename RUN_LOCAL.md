# Running this repository locally

This repository contains two independent Python projects. Choose the project you want to run:

1. **Corporate credit rating** (repository root): trains a LoRA adapter and serves a Gradio UI.
2. **Credit-risk fine-tuning workbench** (`credit-risk-finetuning/`): provides a local dashboard and a Docker-based reviewed SQL application.

Both MLX training paths require an Apple Silicon Mac. The workbench itself and its offline tests can run without training a model.

## Prerequisites

- Git
- Python 3.12 or 3.13
- Apple Silicon and sufficient free disk space for MLX models if you will train or serve a model
- [`uv`](https://docs.astral.sh/uv/) for the workbench project
- Docker Desktop for the reviewed SQL application

Clone the repository, enter it, and then follow one of the paths below.

## Option A: run the fine-tuning workbench

This is the quickest way to open the local dashboard. From the repository root:

```sh
cd credit-risk-finetuning
uv sync --frozen --extra dev --extra training --extra rag --extra documents --extra data
uv run --no-sync python -m credit_risk.workbench.server
```

Open <http://127.0.0.1:8090> in a browser. Stop the server with `Ctrl+C`.

To verify the installation without external services:

```sh
uv run --no-sync pytest -q
```

## Option B: run the reviewed SQL application

From the repository root:

```sh
cd credit-risk-finetuning
uv sync --frozen --extra dev --extra rag --extra documents --extra training
uv run python scripts/setup_local.py
uv run python scripts/make_fixture.py
docker compose up -d --build api data-service qdrant
```

Open <http://127.0.0.1:8080/review>. Sign in with the reviewer token stored in:

```sh
cat .reviewer-token
```

Keep `.env` and `.reviewer-token` private and do not commit them.

Check or stop the services with:

```sh
docker compose ps
docker compose logs -f api
docker compose down
```

The API expects native oMLX at `host.docker.internal:9905` for model-backed responses. The review workflow can be brought up independently, but model calls require oMLX and a compatible model.

## Option C: run the corporate credit-rating project

Run these commands from the repository root (not from `credit-risk-finetuning/`):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Generate a small development dataset first. Increase `--n` for a real training run:

```sh
python scripts/generate_data.py --n 3000 --out ./data
```

Train the LoRA adapter:

```sh
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-8bit \
  --train \
  --data ./data \
  --iters 1000 \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --mask-prompt \
  --adapter-path ./adapters
```

For local iteration, serve the base model with the adapter directly:

```sh
mlx_lm.server \
  --model mlx-community/Qwen2.5-7B-Instruct-8bit \
  --adapter-path ./adapters \
  --port 8080
```

In a second terminal, activate the same virtual environment and start the UI:

```sh
source .venv/bin/activate
python app.py --base-url http://localhost:8080/v1
```

Open <http://localhost:7860>. Stop both processes with `Ctrl+C`.

### Optional: evaluate the adapter

```sh
mkdir -p eval_results
python scripts/eval.py --adapter-path ./adapters --out eval_results/tuned.json
python scripts/eval.py --out eval_results/base.json
```

The first run evaluates the fine-tuned adapter; the second provides the base-model comparison.

## Troubleshooting

- **`uv: command not found`**: install `uv`, then open a new terminal.
- **Python version error in `credit-risk-finetuning/`**: use Python 3.12 or 3.13; its package metadata excludes Python 3.14.
- **The Gradio UI cannot reach the model**: ensure `mlx_lm.server` is still running and that `--base-url` uses port `8080`.
- **Port already in use**: stop the existing process or pass `--port` to `app.py`; update the model-server port and `--base-url` together if port 8080 is occupied.
- **First run is slow**: the model must be downloaded before training or serving, and it requires several gigabytes of disk space.

For architecture, data-contract, production-serving, and full training details, see [`README.md`](README.md) and [`credit-risk-finetuning/README.md`](credit-risk-finetuning/README.md).
