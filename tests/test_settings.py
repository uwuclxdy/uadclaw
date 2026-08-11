"""Settings load: blank credentials must fail closed, a mounted secret file must outrank
an env var of the same name (design: "secrets dir with env-var fallback" only holds if the
file is checked first) — and no credential is ever renderable, because a settings failure
reaches an authenticated dashboard user over HTTP.

The render path is not hypothetical: five stage handlers call `get_settings()`, `worker.py`
persists `traceback.format_exc()` into `jobs.failure_reason` plus a line into `log_tail`,
and both columns are exposed on `JobResponse`.
"""

import pytest
from pydantic import SecretStr, ValidationError

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
    "DEEPSEEK_KEY": "deepseek_key",
}
# Distinctive per credential, and pairwise disjoint at the window `_leaked_fragment` scans
# with, so a fragment found in a rendered string names exactly one field.
CANARIES = {
    "POSTGRES_PASSWORD": "pgpw-8f3a1c9e7b5d2046-canary",
    "AUTH_PASSWORD": "auth-0b6e2f9d4a7c1385-canary",
    "SESSION_SECRET": "sess-4d2b8e6a0c1f3597-canary",
    "DEEPSEEK_KEY": "dsk-7e1f9c3b5a8d2064-canary",
}


def _set_required_env(monkeypatch, override: dict[str, str] | None = None):
    values = {**REQUIRED_ENV, **(override or {})}
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _leaked_fragment(secret: str, text: str, *, window: int = 8) -> str | None:
    """The longest run of `secret` at least `window` long that appears verbatim in `text`.

    `secret in text` is not the check to make: pydantic elides the MIDDLE of a rendered
    value, so a leak arrives as a head and a tail (22 of a 35-character key, measured) and
    never as the whole string.
    """
    for size in range(len(secret), window - 1, -1):
        for start in range(len(secret) - size + 1):
            if secret[start : start + size] in text:
                return secret[start : start + size]
    return None


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

    assert settings.postgres_password.get_secret_value() == "from-file"
    assert settings.auth_password.get_secret_value() == "auth-from-file"
    assert settings.session_secret.get_secret_value() == "secret-from-file"


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

    assert settings.postgres_password.get_secret_value() == "from-env"
    assert settings.auth_password.get_secret_value() == "auth-from-env"
    assert settings.session_secret.get_secret_value() == "secret-from-env"


# --- a settings failure is HTTP-readable, so it may not render a credential ---------------------


def test_a_settings_failure_never_renders_another_credential(monkeypatch, tmp_path):
    """The measured leak: one absent credential and pydantic prints the three it had
    already collected. `_env_file=None` on purpose — the repo's own `.env` holds a live
    key, and this test must never be the thing that renders one."""
    for env_name, value in CANARIES.items():
        monkeypatch.setenv(env_name, value)
    monkeypatch.delenv("AUTH_PASSWORD", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_secrets_dir=str(tmp_path), _env_file=None)

    rendered = str(exc_info.value)
    for env_name, value in CANARIES.items():
        leak = _leaked_fragment(value, rendered)
        assert leak is None, f"{env_name} leaked {leak!r} into the rendered ValidationError"
    assert "auth_password" in rendered, "the error still has to name the field that failed"
    assert "input_value" not in rendered


def test_a_rejected_credential_value_is_never_rendered(monkeypatch, tmp_path):
    """The second render shape, and the one a `SecretStr` annotation does NOT cover: for an
    after-validator error pydantic prints the RAW source string, because the mask only
    exists once the value has been validated into one."""
    _set_required_env(monkeypatch, {"AUTH_PASSWORD": " \t "})

    with pytest.raises(ValidationError) as exc_info:
        Settings(_secrets_dir=str(tmp_path), _env_file=None)

    rendered = str(exc_info.value)
    assert "input_value" not in rendered
    assert "must not be empty or whitespace-only" in rendered


@pytest.mark.parametrize("env_name", sorted(CANARIES))
def test_every_credential_is_masked_in_a_settings_repr(monkeypatch, tmp_path, env_name):
    """Per field, because `repr`/`str` render all four at once: a logged settings object,
    or pytest's own `--showlocals`, is the same disclosure by another route."""
    for name, value in CANARIES.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_secrets_dir=str(tmp_path), _env_file=None)

    for rendered in (repr(settings), str(settings)):
        leak = _leaked_fragment(CANARIES[env_name], rendered)
        assert leak is None, f"{env_name} leaked {leak!r} into {rendered[:80]!r}"
        assert "**********" in rendered


@pytest.mark.parametrize("env_name", sorted(FIELD_BY_ENV_NAME.keys() - {"DEEPSEEK_KEY"}))
def test_settings_rejects_an_unset_credential(monkeypatch, tmp_path, env_name):
    """Absent has to fail exactly like blank, and name only the field that is absent."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_secrets_dir=str(tmp_path), _env_file=None)

    failing_fields = {error["loc"][0] for error in exc_info.value.errors()}
    assert failing_fields == {FIELD_BY_ENV_NAME[env_name]}


# --- an empty secret FILE must not shadow a set env var ------------------------------------------


@pytest.mark.parametrize("blank", ["", "   \n"])
def test_a_blank_secret_file_does_not_shadow_an_env_credential(monkeypatch, tmp_path, blank):
    """A 0-byte `secrets/<name>` placeholder, written to satisfy `docker compose`, must not
    disable the environment fallback the file secrets source deliberately outranks."""
    (tmp_path / "postgres_password").write_text(blank)
    _set_required_env(monkeypatch, {"POSTGRES_PASSWORD": "pg-from-env"})

    settings = Settings(_secrets_dir=str(tmp_path), _env_file=None)

    assert settings.postgres_password.get_secret_value() == "pg-from-env"


@pytest.mark.parametrize("blank", ["", "   \n"])
def test_a_blank_secret_file_does_not_shadow_the_deepseek_key(monkeypatch, tmp_path, blank):
    """`deepseek_key` takes this the worst: it is allowed to be empty, so a shadowed key
    is not a startup failure — it is a classification stage that refuses to run."""
    (tmp_path / "deepseek_key").write_text(blank)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_KEY", "dsk-from-env")

    settings = Settings(_secrets_dir=str(tmp_path), _env_file=None)

    assert settings.deepseek_key.get_secret_value() == "dsk-from-env"


@pytest.mark.parametrize("field", sorted(FIELD_BY_ENV_NAME.values()))
def test_a_skipped_blank_secret_file_is_logged(monkeypatch, tmp_path, caplog, field):
    """Falling back silently is how a truncated rotation goes unnoticed: an operator empties
    `secrets/auth_password`, a stale `AUTH_PASSWORD` still in the environment becomes the
    live login credential, and before the skip existed that combination refused to start."""
    (tmp_path / field).write_text("")
    _set_required_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_KEY", "dsk-from-env")

    with caplog.at_level("WARNING", logger="uadclaw.settings"):
        Settings(_secrets_dir=str(tmp_path), _env_file=None)

    skipped = [r for r in caplog.records if "ignoring blank secret file" in r.getMessage()]
    assert len(skipped) == 1
    # `args`, not the rendered text: a pre-built f-string would leave args empty, and the
    # house rule is a template plus fields so they stay lazy and queryable.
    assert skipped[0].args[0] == field
    assert field in skipped[0].getMessage()


def test_an_absent_secret_file_logs_nothing(monkeypatch, tmp_path, caplog):
    """The half that rots. An absent file is the normal local case — four warnings every
    run would stop being read — so silence there is the point, and the blank file at the
    end is the positive control that proves this fixture can observe a warning at all.

    The silence leg is unkillable from inside the source by construction, since an absent
    file never reaches the dict the base class returns; widening the guard to warn on every
    present file leaves it green and reds the two ordering tests instead. Measured, so
    nobody re-attempts it: the positive control is what carries this test's weight.
    """
    _set_required_env(monkeypatch)

    with caplog.at_level("WARNING", logger="uadclaw.settings"):
        Settings(_secrets_dir=str(tmp_path), _env_file=None)

    assert [r for r in caplog.records if "secret file" in r.getMessage()] == []

    (tmp_path / "postgres_password").write_text("")
    with caplog.at_level("WARNING", logger="uadclaw.settings"):
        Settings(_secrets_dir=str(tmp_path), _env_file=None)

    assert [r for r in caplog.records if "secret file" in r.getMessage()] != []


def test_a_non_blank_secret_file_still_outranks_the_environment(monkeypatch, tmp_path):
    """The source ORDER is unchanged; only what a blank file contributes changed."""
    (tmp_path / "deepseek_key").write_text("dsk-from-file")
    _set_required_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_KEY", "dsk-from-env")

    settings = Settings(_secrets_dir=str(tmp_path), _env_file=None)

    assert settings.deepseek_key.get_secret_value() == "dsk-from-file"


def test_a_credential_is_a_secret_str(monkeypatch, tmp_path):
    """The type is the guarantee every render path above leans on."""
    for name, value in CANARIES.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_secrets_dir=str(tmp_path), _env_file=None)

    for field_name in FIELD_BY_ENV_NAME.values():
        assert isinstance(getattr(settings, field_name), SecretStr), field_name
