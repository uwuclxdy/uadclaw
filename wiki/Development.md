# Development

**The local loop: dependencies, lint, test database, running from source, migrations.**

## Local loop

```sh
uv sync --frozen
uv run ruff check
uv run ruff format --check
uv run pytest
```

`uv sync --frozen` refuses to update `uv.lock`; run `uv sync` (no `--frozen`) after editing `pyproject.toml`'s dependency list and commit the regenerated lockfile. Ruff both lints and formats: `ruff check` covers `E`, `F`, `I`, `B`, `UP` and `SIM`, line length 100.

## Real Postgres, no exceptions

The suite needs a real Postgres reachable at `127.0.0.1:55432` by default: job/lease tests exercise `SKIP LOCKED` semantics, lease contention and crash reclaim, none of which a mock can prove. Bring one up first:

```sh
docker compose up -d --wait postgres
```

That publish comes from `docker-compose.override.yml`, which `docker compose` merges in automatically (never pass `-f docker-compose.yml`, see [Deployment](Deployment)). The override is gitignored, so a fresh clone lacks it. Write it yourself at the repo root:

```yaml
services:
  postgres:
    ports:
      - "127.0.0.1:${POSTGRES_TEST_PORT:-55432}:5432"
```

Loopback only, never `0.0.0.0`: the base compose file publishes no host-facing Postgres port because the deployed topology has none, and this overlay adds back the one thing local tests need. The alternative is to point `UADCLAW_TEST_PG_HOST`/`UADCLAW_TEST_PG_PORT` at a Postgres you already run.

`tests/conftest.py` fails loudly, never skips, when nothing answers at those coordinates: `_ensure_test_database` raises `RuntimeError` naming the host, port, database and user, and the fix command. A skip here would let a suite pass green having proven nothing about the job substrate.

## Per-worker database isolation

`pytest-xdist` runs the suite in parallel (`-n auto --dist loadscope`, set in `pyproject.toml`). Each xdist worker gets its own database, named `{UADCLAW_TEST_DB_PREFIX:-uadclaw_test}_{worker_id}`, created if missing and migrated with `alembic upgrade head` once per worker process (session-scoped fixture, not `Base.metadata.create_all`, so the migration itself is exercised on every run). Each test then truncates every table in that database before running, so tests sharing a worker don't see each other's leftover rows.

The worker id alone (`gw0`, `gw1`, ...) is not enough to isolate two concurrent checkouts of the same repo, since xdist assigns the same worker ids in each. `UADCLAW_TEST_DB_PREFIX` gives a second checkout (a git worktree, for instance) its own database namespace so the two runs can never collide.

## Heavy tests

Tests marked `heavy` run the real extraction chain against a multi-GB local firmware image and are excluded from the default suite. Opt in:

```sh
UADCLAW_HEAVY_TESTS=1 uv run pytest -m heavy
```

Individual heavy test files add their own environment variables for the specific corpus or workdir they need (`UADCLAW_HEAVY_APK_DIR`, `UADCLAW_HEAVY_WORKDIR`, `UADCLAW_HEAVY_IMAGE`); each test file's own docstring or header comment states which apply. Several assert exact package counts against a ground-truth manifest at `docs/research/emulator-a16-packages.tsv`. `docs/` is gitignored, so that file is not part of the repo checkout: tests that need it skip with a message naming the path when it's absent, rather than failing outright. Heavy tests never accept "it didn't raise" as success; they assert counts.

## Running the app from source

```sh
cp .env.example .env   # then fill in the required credentials
uv run uvicorn uadclaw.asgi:app --reload
```

Needs a Postgres reachable at the coordinates in `.env` and the three required credentials set (`POSTGRES_PASSWORD`, `AUTH_PASSWORD`, `SESSION_SECRET`); see [Configuration](Configuration) for the full settings reference. The worker runs the same way: `uv run python -m uadclaw.worker`.

## Alembic autogenerate

`alembic/env.py` imports `uadclaw.models` for its side effect before reading `target_metadata = Base.metadata`; a model that never gets imported there autogenerates as nothing. Every ORM model has to inherit from `uadclaw.db.Base`, or `alembic revision --autogenerate` silently produces an empty migration that reads as "no changes" rather than an error.

```sh
uv run alembic revision --autogenerate -m "describe the change"
uv run alembic upgrade head
```

Point `POSTGRES_*` at a real database first (the loopback test Postgres from `docker compose up -d --wait postgres` works). `alembic.ini`'s own `sqlalchemy.url` is a placeholder; `env.py` overrides it at runtime from `Settings().database_url`, so a migration always targets the database the app itself would connect to.

## What CI gates

`.github/workflows/ci.yml` runs on push to `mommy` and on every PR. The `check` job runs `uv sync --frozen`, `ruff check`, `ruff format --check` and `uv run pytest -q` against a real `postgres:16-alpine` service container on port 55432, matching this doc's local coordinates so no extra env vars are needed in CI. A separate `docker-build` job runs `docker build .`.

`requires-python = ">=3.12.13"` is the only interpreter request in the repo, and it is a security floor rather than tidiness. A looser `>=3.12` let CI resolve the runner's preinstalled 3.12.3, where `ipaddress` classifies a 6to4 address as public and the SSRF gate in [Corroboration](Corroboration) would have fetched the tunnelled cloud metadata address. There is deliberately no `.python-version`: an exact patch pin there is a hard request that the `python:3.12-slim` builder cannot satisfy the day Docker Hub rebases the tag, which is what broke `docker-build` on 2026-08-14 when the image moved from 3.12.13 to 3.12.14. The floor alone accepts any patch at or above the boundary.

## Architecture rules before touching code

A handful of safety-critical invariants span multiple modules. Read the owning page before changing code near them:

- the removal floor a model may raise and never lower: [Rule-Ladder](Rule-Ladder)
- classification as its own job kind, separate from the deterministic firmware walk: [Jobs-and-Worker](Jobs-and-Worker), [Classification](Classification)
- `dependencies`/`neededBy` coming only from the corpus graph or a human, never the model: [Facts-and-Corpus](Facts-and-Corpus)
- the corroboration judge's fabricated-citation refusal: [Corroboration](Corroboration)
- unpacking dispatch on file bytes, never on OEM or filename: [Unpacking](Unpacking)
- per-OEM acquisition posture and index ordering: [Firmware-Drivers](Firmware-Drivers)
- the human gate and append-only decision log: [Triage-and-Emission](Triage-and-Emission)

`CLAUDE.md` at the repo root is the fuller list, kept current as the navigation index; this page links the parts that bite hardest during ordinary development.

## Related

[Configuration](Configuration) is the full settings reference. [Deployment](Deployment) covers the same containers built and hardened for production.
