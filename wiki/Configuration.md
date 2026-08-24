# Configuration

**Every setting uadclaw reads, how it loads, and what it gates.**

## How settings load

`src/uadclaw/settings.py` defines one `Settings` object read once and cached by `get_settings()`. Three sources are merged in this order, first hit wins:

1. a mounted secrets directory, `/run/secrets` inside the container (docker secrets, one file per credential, filename matches the setting name lowercased)
2. environment variables, matched case-insensitively to the setting name
3. `.env` in the repo root (copy `.env.example` to start one)

A **blank file** in the secrets directory does not win: it is treated as absent and the loader falls through to the environment, logging a warning naming the field. This exists so truncating a credential file mid-rotation cannot silently promote a stale environment variable to live without a trace. An **absent** file behaves the same way, silently, which is what makes a plain local run (no secrets directory at all, values from the environment) work without noise.

`POSTGRES_PASSWORD`, `AUTH_PASSWORD` and `SESSION_SECRET` are refused at startup if blank or whitespace-only. An empty `AUTH_PASSWORD` would make the login check pass for anyone; an empty `SESSION_SECRET` would let anyone forge a session cookie offline. The app will not boot rather than boot open. Validation errors never echo the offending value (`hide_input_in_errors=True`): a bad non-secret setting names the constraint and the fix instead of the value it got.

Five stage handlers call `get_settings()` and a settings failure is HTTP-readable: `worker.py` persists the traceback into `jobs.failure_reason` and a line into `log_tail`, both exposed on `JobResponse`.

## Required to start

Only three settings are validated at process startup and refuse to boot when blank:

| key | why |
|---|---|
| `POSTGRES_PASSWORD` | empty would still connect if Postgres allowed it, but the app refuses regardless |
| `AUTH_PASSWORD` | empty makes the constant-time password check trivially true |
| `SESSION_SECRET` | empty signs every session cookie with an empty HMAC key, forgeable offline |

Every other setting either has a working default or is refused later, at the point something tries to use it.

## Credentials

| key | default | type | what it does |
|---|---|---|---|
| `POSTGRES_PASSWORD` | none, required | secret | Postgres connection password |
| `AUTH_PASSWORD` | none, required | secret | the single login password |
| `SESSION_SECRET` | none, required | secret | HMAC key signing the session cookie |
| `LLM_<ID>_KEY` | `""` | secret | one provider's API key (`LLM_DEEPSEEK_KEY` for the default). Blank boots fine; refused only at the first call (`llm.require_api_key`), naming the file and the environment variable. In docker it is a secret file `secrets/llm_<id>_key` mounted into the worker only, never the web service |
| `BRAVE_KEY` | `""` | secret | the corroboration search API key. Same posture: blank boots fine, refused at the point of use (`brave.require_brave_key`). A blank key means a classification job ends at `corroborate` instead of finishing it |

## Postgres

| key | default | type | what it does |
|---|---|---|---|
| `POSTGRES_HOST` | `postgres` | str | database host |
| `POSTGRES_PORT` | `5432` | int | database port |
| `POSTGRES_USER` | `uadclaw` | str | database user |
| `POSTGRES_DB` | `uadclaw` | str | database name |
| `POSTGRES_CONNECT_TIMEOUT_SECONDS` | `10.0` | float, must be `> 0` | ceiling on one connection attempt, so a routable-but-dead host fails fast instead of hanging every dashboard screen for asyncpg's own 60-second default |

## Auth and session

| key | default | type | what it does |
|---|---|---|---|
| `SESSION_COOKIE_NAME` | `uadclaw_session` | str | the session cookie's name |
| `COOKIE_SECURE` | `false` | bool | set the cookie's `Secure` flag. Plain-HTTP LAN deployment by default; flip to `true` only once the dashboard sits behind TLS |

## Worker and job substrate

| key | default | type | what it does |
|---|---|---|---|
| `SCRATCH_ROOT` | `/scratch` | path | root directory the worker unpacks firmware into |
| `WORKER_POOL_SIZE` | `1` | int, must be `>= 1` | concurrent worker slots in this process |
| `LEASE_STALE_AFTER_SECONDS` | `300.0` | float, must be `> 0` | how long a job/lease heartbeat may go stale before it is reclaimed from a dead worker |
| `HEARTBEAT_INTERVAL_SECONDS` | `30.0` | float, must be `> 0` | how often a held lease's heartbeat refreshes during a running job |
| `LEASE_POLL_INTERVAL_SECONDS` | `0.2` | float, must be `> 0` | how often a lease-acquire attempt retries when scratch is occupied |
| `JOB_CLAIM_POLL_INTERVAL_SECONDS` | `1.0` | float, must be `> 0` | how often an idle worker slot polls for a queued job |
| `FAILURE_RETENTION_BYTES` | `5_000_000_000` | int, must be `> 0` | total bytes of scratch kept for jobs no longer actively owned (failed, orphaned, superseded); oldest evicted first once a new one would push the total over this ceiling. At the default, a failed firmware job's ~15-20 GB scratch is always evicted whole |
| `MAX_JOB_ATTEMPTS` | `5` | int, must be `>= 1` | a job stuck claimed/running past this many claims is parked failed instead of requeued ahead of healthy jobs forever |
| `SWEEP_INTERVAL_SECONDS` | `60.0` | float, must be `> 0` | how often the periodic sweep (stale-job/lease reclaim, retention ceiling) runs on top of the one-time sweep at worker startup |
| `STATS_LOOKBACK_SECONDS` | `86_400.0` | float, must be `> 0` | how far back `/stats` aggregates, ending at now rather than at the last completion |

## Firmware acquisition: per-driver enablement and device lists

| key | default | type | what it does |
|---|---|---|---|
| `DISABLED_FIRMWARE_DRIVERS` | `""` | str, comma-separated | driver names to refuse (e.g. `samsung,oppo`). Empty enables every driver. `get_driver` raises `FirmwareDriverDisabledError` naming the setting for a disabled driver |
| `PIXEL_INDEX_URL` | `https://developers.google.com/android/images` | str | Pixel factory-image index |
| `PIXEL_TERMS_ACK_COOKIE_NAME` | `devsite_wall_acks` | str | the terms-acknowledgement cookie's name |
| `PIXEL_TERMS_ACK_COOKIE_VALUE` | `""` | str | acceptance of Google's factory-image terms, unset by default so nothing accepts a licence on the operator's behalf. Blank means `PixelDriver.list_available` raises `FirmwareTermsNotAcknowledgedError` at the `acquire` stage rather than reporting an empty index |
| `XIAOMI_INDEX_URL` | [the tracker index](https://github.com/XiaomiFirmwareUpdater/miui-updates-tracker/blob/master/data/latest.yml) | str | Xiaomi's only machine-readable index |
| `NOTHING_RELEASES_URL` | `https://api.github.com/repos/spike0en/nothing_archive/releases` | str | Nothing's GitHub releases index, unauthenticated |
| `MOTOROLA_MIRROR_URL` | `https://mirrors.lolinet.com` | str | the h5ai mirror Motorola firmware and its index both hang off |
| `MOTOROLA_DEVICES` | `""` | str, comma-separated | codenames to enumerate (e.g. `rtwo,bronco`). Blank means `MotorolaDriver.list_available` raises at `acquire` naming the setting: lolinet publishes no index, so there is nothing to enumerate without an operator-named list |
| `SAMSUNG_INDEX_URL` | `https://fota-cloud-dn.ospserver.net/firmware` | str | the public Samsung version index |
| `SAMSUNG_MODELS` | `""` | str, comma-separated | models to probe (e.g. `SM-S911U,SM-S928B`) |
| `SAMSUNG_REGIONS` | `""` | str, comma-separated | CSC region codes to probe (e.g. `XAA,EUX`). Order is load-bearing: no Samsung build id carries a date, so "newest" resolves to the last region listed here. Blank in either `SAMSUNG_MODELS` or `SAMSUNG_REGIONS` raises at `acquire`; the two multiply into one request per pair, so a wide grid is a wide crawl |
| `OPPO_CATALOGUE_URL` | `https://roms.danielspringer.at/api/ota.php?latest=1` | str | the third-party Oppo/OnePlus/realme catalogue, read once per job and cached |
| `OPPO_MODELS` | `""` | str, comma-separated `MODEL:REGION` entries | models the catalogue never carried, each naming the endpoint region that resolves it (e.g. `RMX3301:EU,RMX3706:GL`). Optional, additive to the catalogue; a model the catalogue already carries is a no-op with a log line |
| `FIRMWARE_HTTP_TIMEOUT_SECONDS` | `60.0` | float, must be `> 0` | read/connect timeout for firmware HTTP, no total deadline: a slow multi-GB transfer that keeps progressing is not a failure |
| `PAYLOAD_DUMPER_PATH` | `payload-dumper-go` | str | path to the `payload-dumper-go` binary. A bare name resolves on `PATH`; the worker image vendors it there |
| `MAX_FIRMWARE_ARCHIVE_BYTES` | `32 * 1024**3` (32 GiB) | int | ceiling on one downloaded archive and on any one member unpacked out of it, stopping a mislabelled URL or decompression bomb rather than acting as a disk quota |

## Fact extraction

| key | default | type | what it does |
|---|---|---|---|
| `MAX_APK_PARSE_FAILURE_RATIO` | `0.05` | float, must be in `[0.0, 1.0)` | fraction of one device's APKs allowed to fail parsing before the whole device is refused |

## Corpus, filter, and rule ladder

| key | default | type | what it does |
|---|---|---|---|
| `UPSTREAM_LIST_PATH` | `/data/uad_lists.json` | path | the operator-supplied upstream list the filter stage reads to know what is already carried. The worker image mounts `./data` read-only at `/data`. `load_upstream_list` refuses a missing or empty file rather than treating the whole corpus as new, and records the copy's sha256 and mtime on every row it decides |

## Classification (LLM providers)

The deterministic pipeline (`acquire` through `rule_ladder`) needs none of these; a `classification` job needs them. The old `DEEPSEEK_*` settings are gone (2026-08-24 provider generalization): the per-provider values live in the provider table, and only the three global client knobs remain flat.

| key | default | type | what it does |
|---|---|---|---|
| `LLM_PROVIDERS` | `{}` | JSON object, one entry per provider id | the provider table. Each entry carries `base_url` (the OpenAI-format endpoint, never `/anthropic`: the cost measurement reads `usage.prompt_cache_hit_tokens`, which the Anthropic wire format does not carry), `model`, `thinking`, `max_tokens` (output ceiling per call, covering reasoning plus the JSON body — too small returns `finish_reason="length"` with empty content, indistinguishable from the documented empty-content bug from the envelope alone) and `max_concurrency` (in-flight requests; the account-wide ceiling has no documented RPM/TPM). A job's `provider` param names one of the ids, validated at creation; jobs that name none use `LLM_DEFAULT_PROVIDER`. Both containers read it — the web process validates at creation and the worker spends against it. The `.env.example` deepseek entry reproduces the old defaults verbatim. `thinking: false` sends the DeepSeek-specific `{"thinking": {"type": "disabled"}}` wire field, so settings load refuses it on any entry that is not deepseek (by id `deepseek` or a base_url on `api.deepseek.com`), naming the entry; deepseek entries keep the flag in either state |
| `LLM_DEFAULT_PROVIDER` | `deepseek` | str | the provider id a classification job uses when its params name none; the resolved id is stored on the job row at creation |
| `LLM_MAX_ATTEMPTS` | `3` | int, must be `>= 1` | wire retries inside one call: 3 means one request plus two on a 429/500/503 or a transport error. Global across providers: the retry layer is one |
| `LLM_RETRY_BACKOFF_SECONDS` | `2.0` | float, must be `> 0` | first backoff, doubled per attempt |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `300.0` | float, must be `> 0` | per-request read/connect deadline, generous because the API can hold a connection open up to 10 minutes before inference starts |
| `CLASSIFICATION_MAX_PACKAGES` | `500` | int, must be `>= 1` | ceiling on packages one classification job may call the API for. A job's own params may lower this, never raise it |
| `CLASSIFICATION_MAX_CALLS_PER_PACKAGE` | `3` | int, must be `>= 1` | total billed requests one package may cost, counting every wire retry. `package_classification.attempts` stores this count |

## Corroboration (Brave search and judge)

A classification job ends at `corroborate`; these settings gate finishing one.

| key | default | type | what it does |
|---|---|---|---|
| `BRAVE_SEARCH_URL` | `https://api.search.brave.com/res/v1/web/search` | str | the search endpoint |
| `BRAVE_RESULT_COUNT` | `10` | int, must be `>= 1` | the API's own `count` parameter |
| `CORROBORATION_SOURCES_PER_PACKAGE` | `10` | int, must be `>= 1` | merged results fetched and shown to the judge per package; each one costs a page fetch |
| `BRAVE_REQUEST_TIMEOUT_SECONDS` | `15.0` | float, must be `> 0` | read/connect deadline for one search call |
| `CORROBORATION_MAX_QUERIES_PER_JOB` | `500` | int, must be `>= 1` | queries one corroboration job may spend, a job-level ceiling set against whatever plan the operator holds |
| `BRAVE_MAX_CONCURRENCY` | `8` | int, must be `>= 1` | searches in flight at once across the whole job |
| `CORROBORATION_SEARCH_TTL_DAYS` | `30.0` | float, must be `> 0` | how long a package's cached search rows stand before it re-searches, so a re-classification does not re-spend the search quota |
| `CORROBORATION_MAX_PACKAGES` | `500` | int, must be `>= 1` | packages one corroboration job may judge; a job's params may lower it, never raise it |
| `CORROBORATION_MAX_CALLS_PER_PACKAGE` | `3` | int, must be `>= 1` | total billed judge requests one package may cost, counting every wire retry |
| `PAGE_FETCH_TIMEOUT_SECONDS` | `10.0` | float, must be `> 0` | read/connect deadline for one page fetch operation |
| `PAGE_FETCH_DEADLINE_SECONDS` | `30.0` | float, must be `> 0` | wall-clock ceiling on one page fetch, redirects and body read included: httpx's timeout restarts on every read, so a trickling response would otherwise never trip it |
| `PAGE_FETCH_MAX_BYTES` | `4 * 1024 * 1024` (4 MiB) | int, must be `>= 1` | hard ceiling on one page body, enforced during streaming. A body past it is truncated, not refused |
| `PAGE_TEXT_MAX_CHARS` | `4000` | int, must be `>= 1` | characters of extracted text kept per source for the judge and the stored row |
| `PAGE_FETCH_MAX_CONCURRENCY` | `8` | int, must be `>= 1` | concurrent page fetches across the whole job |
| `PAGE_FETCH_MAX_REDIRECTS` | `3` | int, must be `>= 0` | redirect hops one page fetch may take, walked by hand so every hop's scheme is checked |

## Branch emission

Blank by default and not refused at startup, unlike the three required credentials: the deterministic pipeline plus triage boots with no clone configured, and only a `branch_emission` job needs one.

| key | default | type | what it does |
|---|---|---|---|
| `UPSTREAM_REPO_PATH` | `""` | str | the operator's own clone of the upstream repo (a fork is fine). Blank means `stages.branch_stage` refuses at the point of use, naming this setting. In docker, `docker-compose.yml` bind-mounts the clone read-write at `/upstream` into the worker only |
| `UPSTREAM_REPO_LIST_PATH` | `resources/assets/uad_lists.json` | str, POSIX-spelled | where `uad_lists.json` lives inside that clone. A fork that moved it repoints this |
| `UPSTREAM_BASE_REF` | `main` | str | the ref a branch is cut from |
| `EMISSION_BRANCH_PREFIX` | `uadclaw` | str | the first segment of every emitted branch name (rest is vendor + job id). Validated at load: must be non-empty, no surrounding whitespace, no leading dash, no whitespace inside |
| `PIPELINE_COMMIT_SHA` | `""` | str | the commit of this pipeline that produced an emission, for the disclosure upstream's CONTRIBUTING requires. Blank refuses emission rather than opening a PR with no provenance. The worker image carries no `.git`, so this arrives from the deploying checkout: `PIPELINE_COMMIT_SHA=$(git rev-parse HEAD) docker compose up -d` |
| `EMISSION_BUNDLE_BASE_URL` | `""` | str | optional public base URL for the evidence bundles, rendered as a link in the PR body. Blank leaves the body explaining how to regenerate them instead of linking |

## Compose-only variables

These are read by `docker-compose.yml` for interpolation, never by `Settings`. Setting them in `.env` or the shell has no effect on the app itself, only on how compose assembles the container.

| key | default | what it does |
|---|---|---|
| `WEB_BIND_ADDR` | `0.0.0.0` | pins the web service's published port to one LAN interface instead of every interface |
| `WORKER_SCRATCH_DIR` | `/mnt/ssd-1/scratch` | host directory bind-mounted as the worker's `/scratch`. Must exist and be writable by uid 65532 before the worker starts: `sudo mkdir -p /mnt/ssd-1/scratch && sudo chown 65532:65532 /mnt/ssd-1/scratch` |
| `UPSTREAM_REPO_DIR` | `/mnt/ssd-1/upstream` | host directory bind-mounted read-write as the worker's `/upstream`. Same ownership requirement as the scratch dir; see [Deployment](Deployment) |
| `PIPELINE_COMMIT_SHA` | unset | passed through into the worker container's environment, where `Settings` reads it as `pipeline_commit_sha` above |

`docker-compose.override.yml` (gitignored, local dev/test only) adds one more: `POSTGRES_TEST_PORT`, defaulting to `55432`, the loopback port Postgres publishes for host-side `pytest` and `alembic`.

## Related

[Deployment](Deployment) covers the compose stack these settings configure. [Development](Development) covers the local loop and test database settings. [Rule-Ladder](Rule-Ladder), [Classification](Classification) and [Corroboration](Corroboration) cover what the stage-specific settings above actually change in pipeline behavior.
