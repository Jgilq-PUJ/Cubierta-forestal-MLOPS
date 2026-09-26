# One Dockerfile, two targets: `jupyter` (training) and `api` (inference).
#
# Both targets install from the SAME cubierta-forestal/pyproject.toml + uv.lock,
# so the scikit-learn version that serialises a model in Jupyter is exactly the
# one that deserialises it in the Inference API.
#
# Build context is the repository root:
#   docker compose build            (compose sets the target per service)
#   docker build --target api .     (manual)

# --------------------------------------------------------------------------- #
# Base with uv
# --------------------------------------------------------------------------- #
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv-base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY cubierta-forestal/pyproject.toml cubierta-forestal/uv.lock ./

# --------------------------------------------------------------------------- #
# Dependencies (one venv per target, resolved from the same lockfile)
# --------------------------------------------------------------------------- #
FROM uv-base AS deps-api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --group api

FROM uv-base AS deps-jupyter
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --group jupyter

# --------------------------------------------------------------------------- #
# Runtime · Inference API (s6)
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS api

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -m -s /bin/bash app

WORKDIR /app
COPY --from=deps-api /app/.venv /app/.venv
COPY cubierta-forestal/src/ /app/src/

RUN chown -R 1000:1000 /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "cubierta_forestal.infrastructure.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------------------------------------- #
# Runtime · JupyterLab (s4, training)
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS jupyter

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/src" \
    UV_PROJECT_ENVIRONMENT="/app/.venv" \
    UV_LINK_MODE=copy

RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -m -s /bin/bash app

# uv stays available inside the dev container: `uv add <package>` updates the
# bind-mounted pyproject.toml / uv.lock and syncs the same venv already on PATH.
COPY --from=uv-base /usr/local/bin/uv /usr/local/bin/uv

WORKDIR /workspace
COPY --from=deps-jupyter /app/.venv /app/.venv
COPY cubierta-forestal/pyproject.toml cubierta-forestal/uv.lock /workspace/
COPY cubierta-forestal/src/ /workspace/src/
COPY services/jupyter/notebooks/ /workspace/notebooks/

RUN chown -R 1000:1000 /workspace /app

USER app
EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.root_dir=/workspace"]
