"""Settings load: blank credentials must fail closed, and a mounted secret file must
outrank an env var of the same name (design: "secrets dir with env-var fallback" only
holds if the file is checked first)."""

import pytest
from pydantic import ValidationError

from uadclaw.settings import Settings

REQUIRED_ENV = {
    "POSTGRES_PASSWORD": "pw",
    "AUTH_PASSWORD": "pw",
    "SESSION_SECRET": "pw",
}
FIELD_BY_ENV_NAME = {
    "POSTGRES_PASSWORD": "postgres_password",
    "AUTH_PASSWORD": "auth_password",
    "SESSION_SECRET": "session_secret",
}


def _set_required_env(monkeypatch, override: dict[str, str] | None = None):
    values = {**REQUIRED_ENV, **(override or {})}
    for key, value in values.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("env_name", ["POSTGRES_PASSWORD", "AUTH_PASSWORD", "SESSION_SECRET"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_settings_rejects_blank_credential(monkeypatch, tmp_path, env_name, blank):
    _set_required_env(monkeypatch, {env_name: blank})
    with pytest.raises(ValidationError) as exc_info:
        Settings(_secrets_dir=str(tmp_path / "no-secrets-here"))

    # Not just "something raised": the two other required fields are valid ("pw"), so the
    # ONLY field allowed to fail is the blank one under test. A guard that fails closed for
    # the wrong reason (e.g. rejecting every value) would satisfy a bare `raises` too.
    failing_fields = {error["loc"][0] for error in exc_info.value.errors()}
    assert failing_fields == {FIELD_BY_ENV_NAME[env_name]}


def test_settings_prefers_secrets_dir_file_over_env(monkeypatch, tmp_path):
    (tmp_path / "postgres_password").write_text("from-file")
    (tmp_path / "auth_password").write_text("auth-from-file")
    (tmp_path / "session_secret").write_text("secret-from-file")
    _set_required_env(
        monkeypatch,
        {
            "POSTGRES_PASSWORD": "from-env",
            "AUTH_PASSWORD": "auth-from-env",
            "SESSION_SECRET": "secret-from-env",
        },
    )

    settings = Settings(_secrets_dir=str(tmp_path))

    assert settings.postgres_password == "from-file"
    assert settings.auth_password == "auth-from-file"
    assert settings.session_secret == "secret-from-file"


def test_settings_falls_back_to_env_when_secrets_dir_absent(monkeypatch, tmp_path):
    _set_required_env(
        monkeypatch,
        {
            "POSTGRES_PASSWORD": "from-env",
            "AUTH_PASSWORD": "auth-from-env",
            "SESSION_SECRET": "secret-from-env",
        },
    )

    settings = Settings(_secrets_dir=str(tmp_path / "does-not-exist"))

    assert settings.postgres_password == "from-env"
    assert settings.auth_password == "auth-from-env"
    assert settings.session_secret == "secret-from-env"
