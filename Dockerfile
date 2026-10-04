# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

COPY demo/sample.jsonl /data/corpus/sample.jsonl
RUN /app/.venv/bin/python -c "from pathlib import Path; from findex.corpus import iter_documents; from findex.index import build_index; from findex.store import save; index = build_index(iter_documents(Path('/data/corpus/sample.jsonl')), positions=True); save(index, Path('/data/index.json')); print(index)"

FROM python:3.13-slim-bookworm AS runtime

RUN groupadd --system --gid 1000 findex \
    && useradd --system --gid 1000 --uid 1000 --home-dir /home/findex --create-home findex

WORKDIR /app
COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /data /data
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH=/data/index.json \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER findex
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health')"

ENTRYPOINT ["/entrypoint.sh"]
