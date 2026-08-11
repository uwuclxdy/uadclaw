"""App configuration.

Reads from a mounted Docker-secrets directory (`/run/secrets`) first, falling back to
environment variables (and a local `.env` for dev) of the same name. `secrets_dir` is a
no-op when the directory does not exist, which is what makes the fallback work.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir="/run/secrets",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "uadclaw"
    postgres_password: str
    postgres_db: str = "uadclaw"

    # Auth: single credential, single-user login (LAN-only deployment).
    auth_password: str
    session_secret: str
    session_cookie_name: str = "uadclaw_session"
    # Plain-HTTP LAN-only deployment by default; flip on if ever put behind TLS.
    cookie_secure: bool = False

    # Worker / job substrate (task 2).
    scratch_root: Path = Path("/scratch")
    worker_pool_size: int = 1
    # How long a job/lease heartbeat may go stale before it is reclaimed from a dead worker.
    lease_stale_after_seconds: float = 300.0
    # How often a held lease's heartbeat is refreshed while a job runs.
    heartbeat_interval_seconds: float = 30.0
    # How often a lease-acquire attempt is retried while scratch is occupied.
    lease_poll_interval_seconds: float = 0.2
    # How often an idle worker slot polls for a queued job.
    job_claim_poll_interval_seconds: float = 1.0
    # Total bytes of scratch artifacts kept on disk for jobs no longer actively owned
    # (failed, orphaned, or a superseded attempt); oldest evicted first once a new one
    # would push the total over this ceiling.
    #
    # NOTE at this default a failed FIRMWARE_ANALYSIS job is always evicted whole: one peaks
    # near 15 GB (a 3.5 GB archive, its partition images, and the extracted tree), so the
    # "keep failed artifacts for debugging" intent never applies to that kind. Raise this
    # past a single job's peak if post-mortem access to a failed unpack is wanted.
    failure_retention_bytes: int = 5_000_000_000
    # A job stuck CLAIMED/RUNNING past this many claims is parked FAILED instead of
    # requeued forever ahead of healthy jobs (claim ordering is FIFO by created_at, so an
    # endlessly-reclaimed job would otherwise jump the queue on every reclaim).
    max_job_attempts: int = 5
    # How often the periodic sweep (stale-job/lease reclaim, retention ceiling) runs while
    # the worker is up, on top of the one-time sweep at startup.
    sweep_interval_seconds: float = 60.0
    # How far back /stats looks when aggregating. Ends at "now", not at the last time
    # something finished, so a wedged system (nothing finishing) still reads accurately.
    stats_lookback_seconds: float = 86_400.0

    # Firmware acquisition (task 3).
    # Comma-separated driver names to refuse. Kept a plain string rather than a set because
    # pydantic-settings parses a collection-typed field as JSON from the environment, which
    # is a worse operator experience for "turn Samsung off": DISABLED_FIRMWARE_DRIVERS=samsung.
    disabled_firmware_drivers: str = ""
    pixel_index_url: str = "https://developers.google.com/android/images"
    # The Pixel factory index is behind a client-side terms wall. Acceptance of Google's
    # image terms is an act by the operator, so it is configuration and starts UNSET: a bare
    # fetch answers HTTP 200 with prose and zero download links, which is why the driver
    # refuses to fetch at all until this is set rather than reporting "no builds".
    pixel_terms_ack_cookie_name: str = "devsite_wall_acks"
    pixel_terms_ack_cookie_value: str = ""
    # Xiaomi (task 11). XiaomiFirmwareUpdater's tracker repo, whose `data/latest.yml` is the
    # only machine-readable index of Xiaomi's own CDN that exists; Xiaomi publishes none.
    xiaomi_index_url: str = (
        "https://raw.githubusercontent.com/XiaomiFirmwareUpdater/"
        "miui-updates-tracker/master/data/latest.yml"
    )
    # Nothing (task 11). Unauthenticated on purpose: decision 8 keeps every GitHub credential
    # out of this stack, and that caps the driver at GitHub's 60 requests/hour/IP.
    nothing_releases_url: str = "https://api.github.com/repos/spike0en/nothing_archive/releases"
    # Motorola (task 11). One host: the h5ai JSON API and the firmware tree both hang off it.
    motorola_mirror_url: str = "https://mirrors.lolinet.com"
    # Motorola codenames to enumerate, comma-separated (`rtwo,bronco`). Required, and a plain
    # string for the same reason `disabled_firmware_drivers` is. lolinet publishes no index
    # document at all — it is a directory tree, 8 year directories over ~26 devices each,
    # every device carrying 10-15 channel directories — so listing "everything on offer"
    # would be thousands of requests per job against a mirror that asks for non-commercial
    # use. The operator names the handful of devices worth tracking instead.
    motorola_devices: str = ""
    # Read/connect timeout for firmware HTTP. No total deadline: a factory zip is multi-GB
    # and a slow-but-progressing transfer is not a failure.
    firmware_http_timeout_seconds: float = 60.0
    # `payload-dumper-go` is not packaged by any distro; the worker image vendors it onto
    # PATH, a dev box may have it anywhere (e.g. ~/go/bin). A bare name is resolved on PATH.
    payload_dumper_path: str = "payload-dumper-go"
    # Ceiling on a single downloaded archive and on any one member unpacked out of it. Not a
    # disk quota: it stops a mislabelled URL or a decompression bomb from filling scratch
    # before anything else notices. A Pixel factory zip is ~3.5 GB, so this leaves headroom
    # for the largest OEM images without leaving the ceiling meaningless.
    max_firmware_archive_bytes: int = 16 * 1024**3

    # Fact extraction (task 4). Fraction of one device's APKs that may fail to parse before
    # the whole device is refused. The measured baseline is 0 of 312 on real Pixel firmware,
    # so any non-zero rate is already worth looking at; this ceiling exists so that one
    # truncated or parser-hostile APK does not cost the other 311, while a systematically
    # broken extraction (wrong bytes, wrong tool) still fails loudly instead of recording a
    # device with a handful of packages.
    max_apk_parse_failure_ratio: float = 0.05

    # Corpus graph / filter / rule ladder (task 5). The upstream `uad_lists.json` the filter
    # reads to know what is already carried. Explicit configuration and a local file on
    # purpose: it decides what the pipeline proposes, so a stage that fetched its own filter
    # input would change what reaches triage between two runs of the same corpus with nothing
    # recorded about why. The worker image mounts ./data read-only at /data; the loader
    # records the copy's sha256 and mtime on every row it decides, and refuses a missing or
    # empty file rather than treating the whole corpus as new.
    upstream_list_path: Path = Path("/data/uad_lists.json")

    # Classification (task 7). Deliberately NOT in `_reject_blank` below, unlike the other
    # three credentials: the whole deterministic half of this pipeline (acquire through
    # rule_ladder, milestone M2) is independently useful and must boot on a box that has no
    # DeepSeek account at all. A blank key is refused at the point of use instead, by
    # `deepseek.require_api_key`, naming the setting and where to put it.
    deepseek_key: str = ""
    # The OpenAI-format endpoint, never `/anthropic`. `usage.prompt_cache_hit_tokens` and
    # `prompt_cache_miss_tokens` are what the cost measurement (task 8) reads, and the
    # Anthropic wire format does not carry them.
    deepseek_base_url: str = "https://api.deepseek.com"
    # Cheapest of the two live models and 5x the concurrency ceiling of `deepseek-v4-pro`
    # (2500 vs 500). The legacy `deepseek-chat`/`deepseek-reasoner` ids no longer resolve.
    deepseek_model: str = "deepseek-v4-flash"
    # Thinking mode is ON by default on v4-flash and its reasoning tokens bill as output.
    # Measured on one identical prompt: default 110 completion tokens (104 reasoning),
    # `reasoning_effort: "low"` 58 (52), `thinking: {"type": "disabled"}` 5 and no reasoning
    # field at all. So `low` is a discount and `disabled` is the actual off switch. Left ON
    # here — the API's own default — because whether disabling it costs description quality
    # is what task 8's measurement answers, and a setting is how both halves get measured.
    deepseek_thinking: bool = True
    # Sized for reasoning PLUS the JSON body, not the body alone: reasoning is charged
    # against this ceiling, and a budget that only covers the answer returns
    # `finish_reason="length"` with EMPTY content — indistinguishable from DeepSeek's
    # documented empty-content bug from the envelope, and with the opposite fix.
    deepseek_max_tokens: int = 4096
    # Concurrency is account-wide across every key (there is no documented RPM or TPM), and
    # the account is shared with whatever else is talking to DeepSeek, so this defaults far
    # below the 2500 ceiling rather than near it.
    deepseek_max_concurrency: int = 4
    # Total attempts per package, not retries on top of one: 3 means one call plus two
    # retries. The retry/backoff layer is the client's alone — the stage must not wrap it in
    # a second one, or the two compound silently.
    deepseek_max_attempts: int = 3
    # First backoff, doubled per attempt. DeepSeek prescribes none and sends no Retry-After.
    deepseek_retry_backoff_seconds: float = 2.0
    # Per-request read/connect deadline. The API holds a connection open for up to 10 minutes
    # before inference starts, so this is generous on purpose.
    deepseek_request_timeout_seconds: float = 300.0
    # How many packages one classification job may call the API for. A ceiling on spend that
    # a job's own params can lower but never raise, so a mis-typed job cannot bill a corpus.
    classification_max_packages: int = 500

    @field_validator("postgres_password", "auth_password", "session_secret")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        # An empty auth_password makes `hmac.compare_digest("", "")` true (anyone logs in);
        # an empty session_secret signs cookies with an empty HMAC key (any cookie payload
        # is forgeable offline, no request to /login required). Fail at startup, not at
        # the first exploited request.
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("worker_pool_size")
    @classmethod
    def _reject_non_positive_pool(cls, value: int) -> int:
        if value < 1:
            raise ValueError("worker_pool_size must be >= 1")
        return value

    @field_validator(
        "lease_stale_after_seconds",
        "heartbeat_interval_seconds",
        "lease_poll_interval_seconds",
        "job_claim_poll_interval_seconds",
        "sweep_interval_seconds",
        "stats_lookback_seconds",
        "firmware_http_timeout_seconds",
        "deepseek_request_timeout_seconds",
        "deepseek_retry_backoff_seconds",
    )
    @classmethod
    def _reject_non_positive_duration(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator(
        "deepseek_max_tokens",
        "deepseek_max_concurrency",
        "deepseek_max_attempts",
        "classification_max_packages",
    )
    @classmethod
    def _reject_non_positive_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("max_apk_parse_failure_ratio")
    @classmethod
    def _reject_out_of_range_ratio(cls, value: float) -> float:
        # 1.0 would let a device record zero packages and still succeed, which is the exact
        # silent failure the ceiling exists to catch.
        if not 0.0 <= value < 1.0:
            raise ValueError("max_apk_parse_failure_ratio must be in [0.0, 1.0)")
        return value

    @field_validator("failure_retention_bytes")
    @classmethod
    def _reject_non_positive_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("failure_retention_bytes must be > 0")
        return value

    @field_validator("max_job_attempts")
    @classmethod
    def _reject_non_positive_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_job_attempts must be >= 1")
        return value

    @property
    def disabled_firmware_driver_names(self) -> frozenset[str]:
        return frozenset(
            name.strip() for name in self.disabled_firmware_drivers.split(",") if name.strip()
        )

    @property
    def motorola_device_names(self) -> tuple[str, ...]:
        """Deduplicated, in the order the operator wrote them, so a crawl is reproducible."""
        seen: dict[str, None] = {}
        for name in self.motorola_devices.split(","):
            if name.strip():
                seen.setdefault(name.strip(), None)
        return tuple(seen)

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # pydantic-settings' default priority is init > env > dotenv > file secrets, i.e.
        # an env var silently beats the mounted secret file. The design is "secrets dir
        # with env-var fallback" — the file must win when present; env is the fallback
        # for when there's no mounted secrets dir at all (local/dev).
        return init_settings, file_secret_settings, env_settings, dotenv_settings


@lru_cache
def get_settings() -> Settings:
    return Settings()
