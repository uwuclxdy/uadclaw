"""App configuration.

Reads from a mounted Docker-secrets directory (`/run/secrets`) first, falling back to
environment variables (and a local `.env` for dev) of the same name. `secrets_dir` is a
no-op when the directory does not exist, which is what makes the fallback work.
"""

from functools import lru_cache

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
