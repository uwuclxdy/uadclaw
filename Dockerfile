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

# `lpunpack` splits a dynamic-partition `super.img` and Debian packages NO form of it: trixie
# ships android-libbase, android-libsparse and android-libext4-utils but no liblp at all
# (`apt-cache policy android-liblp-dev` inside this image finds nothing), so it is compiled
# here rather than installed. `nmeum/android-tools` is the CMake port of the AOSP host tools
# that Arch's own `android-tools` package builds from — Apache-2.0, and `lpunpack` is one of
# its declared install targets, not a hoped-for side effect. That tarball already carries the
# vendored AOSP sources pre-patched, so nothing past this one fetch touches the network and no
# submodule floats.
#
# The sha256 is written TWICE on purpose, and the second one is not redundant: `ADD
# --checksum` is a BuildKit feature and the legacy builder IGNORES it silently — measured with
# an all-zeros digest, `DOCKER_BUILDKIT=0` builds clean and `DOCKER_BUILDKIT=1` fails with
# `digest mismatch`. A GitHub release asset is mutable, `deploy.sh` builds over ssh on a host
# whose builder nobody asserts, and nothing in this tree sets `DOCKER_BUILDKIT`, so the
# `sha256sum -c` below is what actually holds the pin on the legacy builder. Keep the two
# digests equal.
#
# Only the `lpunpack` target is built. The project also produces adb, fastboot, e2fsdroid and
# friends, and a full `make` costs minutes for binaries nothing in this pipeline runs — but
# the CONFIGURE step still probes for every one of their dependencies, which is why the -dev
# list is wider than what lpunpack itself links. Bundled fmt rather than Debian's libfmt so
# the binary needs no `libfmt.so` in the runtime image; everything it does link (libz,
# libstdc++, libgcc_s, libm, libc) python:3.12-slim already ships. The builder IS the runtime
# image for the same reason payload-dumper's tracks the runtime's Debian release — a builder
# libc newer than the runtime's fails at exec inside a firmware job, not at build time — and
# it is spelled as the identical base rather than a pinned Debian so that the invariant cannot
# drift when python:3.12-slim rebases onto the next Debian release. Stripped: 9.0 MB of
# Release build with debug info is 1.7 MB without.
FROM python:3.12-slim AS lpunpack-builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ninja-build \
      pkg-config \
      zlib1g-dev \
      libbrotli-dev \
      liblz4-dev \
      libpcre2-dev \
      libprotobuf-dev \
      libusb-1.0-0-dev \
      libzstd-dev \
      protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:2725d09f892a3a38e534429f47a321f58ecf6a3169caa42c915fb2cb7d46be0e \
    https://github.com/nmeum/android-tools/releases/download/37.0.0/android-tools-37.0.0.tar.xz \
    /tmp/android-tools.tar.xz
# PATCH_VENDOR off because the release tarball is the already-patched source drop: left on,
# CMake reaches for `git submodule update` in a directory that is not a repository.
RUN mkdir -p /src \
 && echo "2725d09f892a3a38e534429f47a321f58ecf6a3169caa42c915fb2cb7d46be0e  /tmp/android-tools.tar.xz" \
    | sha256sum -c - \
 && tar -xf /tmp/android-tools.tar.xz -C /src --strip-components=1 \
 && cmake -S /src -B /src/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_TOOLS_USE_BUNDLED_FMT=ON \
      -DANDROID_TOOLS_PATCH_VENDOR=OFF \
 && cmake --build /src/build --target lpunpack \
 && strip /src/build/vendor/lpunpack

FROM python:3.12-slim
WORKDIR /app

# The unpacking toolchain. Versions matter here: 7-Zip has NO EROFS handler at all — measured
# 2026-08-11, 7-Zip 26.02 lists Ext and SquashFS only, and pointed at an EROFS image it falls
# through to its gzip reader and lists 1 file where fsck.erofs recovers 105. EROFS always goes
# through fsck.erofs; 7z covers ext4. p7zip 16.02 (the `p7zip-full` transitional package) reads
# neither and is never shipped. Debian trixie ships 7-Zip 25.01.
#
# `lpunpack` (dynamic `super.img` partitions) comes from the builder stage above, which is
# what unblocks the Motorola driver: its chain is zip -> sparsechunk set -> simg2img ->
# super.img -> lpunpack, and lpunpack was the one link this image could not install. No
# driver is disabled on account of a missing tool now; `uadclaw.unpack` still raises a named
# MissingToolError rather than failing obscurely if one goes missing. Pixel, Xiaomi and
# Nothing need none of it: a Pixel factory zip carries raw ext4 images directly, a Xiaomi
# recovery ROM goes through payload-dumper-go, and a Nothing archive is a 7z of finished
# partition images.
#
# `simg2img` comes from android-sdk-libsparse-utils below and IS present, which is what the
# Motorola sparsechunk rebuild needs before lpunpack ever runs.
#
# `git` is the odd one out: nothing unpacks with it. The branch-emission stage shells out to it
# against a local clone the operator mounts, and it neither pushes nor authenticates — there is
# no GitHub credential in this stack, a human pushes the branch by hand.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      7zip \
      erofs-utils \
      android-sdk-libsparse-utils \
      git \
 && rm -rf /var/lib/apt/lists/*
COPY --from=payload-dumper /go/bin/payload-dumper-go /usr/local/bin/payload-dumper-go
COPY --from=lpunpack-builder /src/build/vendor/lpunpack /usr/local/bin/lpunpack

# Every tool this image shells out to — `uadclaw.unpack`'s five plus `git` for branch emission —
# resolved and link-checked at BUILD time. A tool this loop does not NAME is unasserted however
# many other lines install it, and the dev box carries all of them, so a tool missing from the
# image passes every local run and fails inside a job: a tool added above gets added here in the
# same change. The gap this closes is the lpunpack builder's eleven -dev packages: CMake probes
# all of them, so an android-tools release that makes one an lpunpack-side dependency gives
# the binary a NEEDED entry this slim runtime has no library for — the image still builds, and
# the failure is a dynamic-loader error the first time a Motorola job reaches its `super.img`.
# `/usr/bin/7z` is a `#!/bin/sh` wrapper (its ELF is `7zz`), so for that one `command -v` is
# the whole check and the ldd half is a deliberate no-op.
# erofs-utils >= 1.8.5 is a floor, not a nicety: before its fragment-cache fix a real
# extraction measured 362.3s and after it 20.8s, and a distro-frozen older build extracts
# correctly but reads as a mysteriously slow job. The worker image ships 1.8.6 (trixie).
RUN set -eu; \
    for tool in 7z fsck.erofs simg2img lpunpack payload-dumper-go git; do \
      resolved="$(command -v "$tool")" || { echo "missing tool: $tool" >&2; exit 1; }; \
      if [ "$tool" = fsck.erofs ]; then \
        version="$(fsck.erofs --version | awk '{print $4}')"; \
        if [ "$(printf '%s\n' "$version" 1.8.5 | sort -V | head -n1)" != 1.8.5 ]; then \
          echo "fsck.erofs too old: $version, need >= 1.8.5" >&2; \
          exit 1; \
        fi; \
      fi; \
      if ldd "$resolved" 2>/dev/null | grep -q 'not found'; then \
        echo "unresolved shared libraries in $resolved:" >&2; \
        ldd "$resolved" >&2; \
        exit 1; \
      fi; \
      echo "ok: $tool -> $resolved"; \
    done

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
