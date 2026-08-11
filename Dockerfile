FROM python:3.12-slim AS builder
# Pin the uv image — `:latest` defeats build-time reproducibility even though `uv.lock`
# pins the Python dep graph. Bump the tag (or move to a sha256 digest) deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY . .
RUN uv sync --frozen --no-dev

# `payload-dumper-go` opens A/B OTA `payload.bin` and is packaged by no distribution, so it
# is built from the pinned module version verified on the dev box rather than pulled as a
# release asset. CGO stays ON and `liblzma-dev` is required: two of its dependencies (go-xz,
# gozstd) are cgo wrappers and the build fails outright without either. The result links
# only against libc, so it drops into the slim runtime image as-is; the builder tracks the
# runtime's Debian release so that libc is never the newer of the two.
FROM golang:1.26-trixie AS payload-dumper
RUN apt-get update \
 && apt-get install -y --no-install-recommends liblzma-dev \
 && rm -rf /var/lib/apt/lists/*
RUN go install github.com/ssut/payload-dumper-go@v0.0.0-20241120142751-a51234eaead2

FROM python:3.12-slim
WORKDIR /app

# The unpacking toolchain. Versions matter here: 7-Zip >= 24 reads BOTH ext4 and EROFS, which
# is what collapses selective extraction to a single tool — p7zip 16.02 (the `p7zip-full`
# transitional package) reads neither. Debian trixie ships 7-Zip 25.01.
#
# `lpunpack` (dynamic `super.img` partitions) is deliberately absent: no Debian package
# provides it. Nothing on the Pixel path needs it — a Pixel factory zip carries raw ext4
# partition images directly, measured 2026-08-11 — and `uadclaw.unpack` raises a named
# MissingToolError naming the tool if a super image ever reaches this image. The OEMs that
# do ship super images arrive with task 11, which is when this needs solving.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      7zip \
      erofs-utils \
      android-sdk-libsparse-utils \
 && rm -rf /var/lib/apt/lists/*
COPY --from=payload-dumper /go/bin/payload-dumper-go /usr/local/bin/payload-dumper-go

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
