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
    )
    @classmethod
    def _reject_non_positive_duration(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
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
