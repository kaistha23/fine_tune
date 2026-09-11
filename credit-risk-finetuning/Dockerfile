FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra rag --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra rag && \
    addgroup --system app && adduser --system --ingroup app --home /app app && \
    chown -R app:app /app

USER app

CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8080"]

FROM runtime AS tests
USER root
RUN uv sync --frozen --extra rag --extra dev --extra documents
COPY compose.yaml ./compose.yaml
COPY tests ./tests
COPY scripts ./scripts
COPY data/evaluation ./data/evaluation
RUN chown -R app:app /app
USER app
CMD ["python", "-m", "pytest", "-q"]
