FROM python:3.12-slim AS builder
# Pin the uv image — `:latest` defeats build-time reproducibility even though `uv.lock`
# pins the Python dep graph. Bump the tag (or move to a sha256 digest) deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app

# UID 65532 is the distroless / nonroot convention — outside the typical 1000-1001 host range
# so bind-mounted files on the deploy host don't appear owned by the operator's own user.
RUN groupadd --system --gid 65532 app \
 && useradd  --system --uid 65532 --gid 65532 --create-home --home-dir /home/app app

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/tmp/cache \
    UV_CACHE_DIR=/tmp/cache

# Pre-create the worker's scratch mount point owned by the non-root user, so docker's
# "populate a fresh named volume from the image directory" behaviour hands it over already
# writable by UID 65532 instead of root-owned. Unused by the web service.
RUN mkdir -p /scratch && chown app:app /scratch

USER app
EXPOSE 8000

# Pure-Python check — `curl` is not in `python:3.12-slim`. Exit non-zero on any failure.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Web entrypoint by default; the worker service overrides `command:` (and disables this
# HEALTHCHECK, since it never listens on 8000) in docker-compose.yml. One image, two roles.
CMD ["python", "-m", "uvicorn", "uadclaw.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
