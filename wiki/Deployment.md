# Deployment

**Docker Compose stack: Postgres, the web dashboard, and a worker, built from one Dockerfile.**

## The stack

`docker-compose.yml` defines three services and no explicit network, so compose creates its default project network and all three join it.

| service | image | role |
|---|---|---|
| `postgres` | `postgres:16-alpine` | the database. No published port: internal-network-only, LAN exposure goes through `web` |
| `web` | built from `Dockerfile` | the dashboard, `uvicorn uadclaw.asgi:app` on port 8000 |
| `worker` | same image as `web`, `command: ["python", "-m", "uadclaw.worker"]` | claims and runs pipeline jobs |

Both `web` and `worker` are `restart: unless-stopped`: a reclaim sweep runs inside the worker process, so a process that dies and stays dead never recovers a job or lease it was holding without the container restarting.

```sh
docker compose up -d --wait postgres
docker compose run --rm web alembic upgrade head
docker compose up -d
```

The dashboard listens at `http://localhost:8000` (or the interface `WEB_BIND_ADDR` names). Login is the single `AUTH_PASSWORD`.

## Never pass `-f` to `docker compose`

Plain `docker compose` merges `docker-compose.override.yml` automatically when it is present. That file is gitignored and adds the test-Postgres loopback publish (`127.0.0.1:${POSTGRES_TEST_PORT:-55432}`) the whole DB-backed test suite runs against. Passing `-f docker-compose.yml` explicitly recreates services without the override and drops that publish, which fails the local suite against a healthy container with nothing listening on the port. Every command in this doc is plain `docker compose ...`; a command block says so explicitly on the rare occasion it isn't.

## Docker secrets

Five credentials are docker secrets: files under `./secrets/`, mounted read-only at `/run/secrets/<name>` inside the containers that declare them.

```sh
mkdir -p secrets
printf 'a long random postgres password' > secrets/postgres_password
printf 'a long random login password'    > secrets/auth_password
printf 'a long random session secret'    > secrets/session_secret
touch secrets/llm_deepseek_key   # blank is fine; the file must exist
touch secrets/brave_key      # blank is fine; the file must exist
```

`postgres_password`, `auth_password` and `session_secret` mount into both `web` and `worker`: `Settings()` refuses to construct with any of the three blank, so both processes need real values just to start, even though the worker serves no HTTP. `llm_deepseek_key` and `brave_key` mount into `worker` only, never `web`: nothing the web service serves calls the model or the search API, so a credential that spends money has no business inside the internet-facing container. Both may be blank content, but the file has to exist, since compose refuses to start a container whose declared secret file is missing. See [Configuration](Configuration) for what a blank `llm_deepseek_key`/`brave_key` gates.

The LLM provider table (`LLM_PROVIDERS`) is configuration, not a credential, and compose interpolates it into BOTH containers from the deploy environment (or the compose `.env`). The web process needs it to validate a classification job's provider at creation — a table only the worker can see means every launch from the dashboard is a 422.

## Bind mounts

| mount | service(s) | purpose |
|---|---|---|
| `./data:/data:ro` | `web`, `worker` | the operator-supplied `uad_lists.json` (`UPSTREAM_LIST_PATH`). The web service reads it too: the triage screen shows nearby existing upstream entries beside each proposal |
| `${WORKER_SCRATCH_DIR:-/mnt/ssd-1/scratch}:/scratch` | `worker` only | scratch space for extracting firmware. A bind mount, not a named volume: a named volume lives under `/var/lib/docker` and a Samsung job's archive alone is 11.5-19.25 GB before it is opened. Must exist and be owned by uid 65532 before the worker starts: `sudo mkdir -p /mnt/ssd-1/scratch && sudo chown 65532:65532 /mnt/ssd-1/scratch` |
| `${UPSTREAM_REPO_DIR:-/mnt/ssd-1/upstream}:/upstream` | `worker` only | the operator's own clone of the upstream repo, read-write, for branch emission. A fork is fine, since the clone is identified by the list file it carries and never by remote. `web` neither emits nor could write anything, given its read-only rootfs |

Setting up the upstream clone:

```sh
sudo git clone https://github.com/<you>/universal-android-debloater-next-generation \
  /mnt/ssd-1/upstream
sudo chown -R 65532:65532 /mnt/ssd-1/upstream
docker compose exec worker git -C /upstream config user.name "<your name>"
docker compose exec worker git -C /upstream config user.email "<your email>"
```

The identity config is required: emission refuses `-c user.name`/`-c user.email` flags and strips the identity environment, so a clone with no configured identity fails the emission naming this step. The worker runs as uid 65532, so running the config through `docker compose exec worker` writes it with the right owner.

Nothing in this stack pushes, fetches, or authenticates against the clone's remote. The deliverable is a local branch plus a PR body recorded on the emission row; a human pushes it by hand.

## Hardening posture

Both application services carry a shared baseline: `cap_drop: [ALL]`, `no-new-privileges:true`, `user: "65532:65532"`, `init: true`, `pids_limit: 200`, `ulimits.nofile` capped at 1024 soft / 2048 hard, and `json-file` logging capped at 10m × 3 files.

`web` runs `read_only: true` on top of that, with `tmpfs` mounts for `/tmp` (64m) and `/run` (8m), and `mem_limit: 512m`. `worker` is deliberately **not** `read_only`: it unpacks multi-GB firmware images and shells out to extraction tools, and `tmpfs` tops out around half the host's RAM, which won't hold an 8 GB image. Its writable surface is the two bind mounts above, not the container filesystem.

`postgres` gets no application hardening block; it runs the stock `postgres:16-alpine` image behind a healthcheck (`pg_isready -U uadclaw -d uadclaw`) that `web` and `worker` both wait on via `depends_on: condition: service_healthy`.

## Dockerfile build stages

Four stages, the last one being the runtime image both `web` and `worker` run from.

| stage | base | produces |
|---|---|---|
| `builder` | `python:3.12-slim` | the app's Python environment via `uv sync --frozen`, dev deps excluded |
| `payload-dumper` | `golang:1.26-trixie` | `payload-dumper-go`, built with `go install` at a pinned module version, CGO on (two of its own deps are cgo wrappers). Not packaged by any distro, so it opens A/B OTA `payload.bin` files by being compiled here rather than pulled from a distro repo |
| `lpunpack-builder` | `python:3.12-slim` (same base as the runtime image, deliberately) | `lpunpack`, compiled from `nmeum/android-tools` (the CMake AOSP-host-tools port Arch's own package builds from). Debian ships no `liblp` at all, so no distro package can provide it. The release tarball's sha256 is checked twice in the Dockerfile: once via `ADD --checksum`, which the legacy (non-BuildKit) docker builder silently ignores, and once via a `sha256sum -c` inside a `RUN`, which is the half that actually holds the pin when `deploy.sh` builds over ssh on a host whose builder isn't asserted |
| final | `python:3.12-slim` | copies the app from `builder` and the two compiled binaries from the stages above, installs `7zip`, `erofs-utils`, `android-sdk-libsparse-utils` and `git` via apt, verifies every shelled-out tool resolves on `PATH` and links cleanly at build time, creates the non-root `app` user (uid/gid 65532), and sets the `HEALTHCHECK` |

The final stage's tool-verification loop checks `7z`, `fsck.erofs`, `simg2img`, `lpunpack`, `payload-dumper-go` and `git`: each must resolve via `command -v` and link with no missing shared libraries (`ldd`). `fsck.erofs` carries a version gate too, `>= 1.8.5` (a real extraction measured 362.3s before that fix and 20.8s after). A tool the loop doesn't name is unasserted, no matter how many other Dockerfile lines install it.

`CMD` runs the web entrypoint by default (`python -m uvicorn uadclaw.asgi:app --host 0.0.0.0 --port 8000`); `worker` in `docker-compose.yml` overrides `command:` and disables the baked-in `HEALTHCHECK`, since the worker never listens on 8000.

## CI

`.github/workflows/ci.yml` runs two jobs on push to `mommy` and on every PR: `check` (uv sync, ruff check, ruff format --check, pytest against a real `postgres:16-alpine` service container) and `docker-build` (`docker build .` against the Dockerfile above). The `docker-build` job proves only the BuildKit path, since GitHub-hosted runners default to it; `deploy.sh` builds over ssh on whatever builder the deploy host has, which is exactly why the Dockerfile pins the `lpunpack` tarball's sha256 twice.

`docker-build` catches things `check` cannot see. It went red on 2026-08-14 when Docker Hub rebased `python:3.12-slim` from 3.12.13 to 3.12.14 and the repo's `.python-version` still requested the older patch exactly. `check` stayed green throughout. See the interpreter-request note in [Development](Development).

## The alembic migration step

Schema changes go through alembic, never `Base.metadata.create_all`. `alembic/env.py` imports `uadclaw.models` for its side effect of registering every mapped table onto `uadclaw.db.Base.metadata`, then reads the database URL from `Settings().database_url` rather than the placeholder in `alembic.ini`, so migrations always target the same database the app itself would connect to.

```sh
docker compose run --rm web alembic upgrade head
```

Run this once after `postgres` is healthy and before starting `web`/`worker` for the first time, and again after pulling any change that adds a migration. Do not run it against the host by publishing Postgres's port on the deployed stack; run it from inside the compose network as shown, or use the loopback publish the local dev override adds (see [Development](Development)).

## deploy.sh

`deploy.sh REMOTE` (or `REMOTE` from `.env`) rsyncs every git-tracked file to `REMOTE_DIR` on the remote host (default `/srv/<repo-basename>`), then runs `docker compose -f COMPOSE_FILE up -d --build` there over ssh.

It prunes files on the remote that were tracked at the previous deploy and have since been removed from git, using a manifest diff (`.deploy-manifest` on the remote) rather than a wholesale sync-and-delete. It never touches named volumes, bind-mounted data directories, or logs: the prune list is built only from what git tracked, and `secrets/`, `data/`, and the scratch/upstream bind mounts are all outside git. `.env`'s `REMOTE`, `REMOTE_DIR` and `COMPOSE_FILE` keys are parsed without shell execution (a tight allowlist regex, no `source`), so a hostile `.env` cannot inject a command.

```sh
./deploy.sh user@host
```

First deploy from a new machine needs the remote host key trusted out of band first: `ssh-keyscan -t ed25519,rsa <host> >> ~/.ssh/known_hosts` after verifying the fingerprint some other way. `deploy.sh` uses `BatchMode=yes`, so it will not prompt for one.

## Related

[Configuration](Configuration) is the full settings reference for everything this stack's environment and secrets configure. [Development](Development) covers running the same containers locally for testing, including the loopback Postgres port `docker-compose.override.yml` adds.
