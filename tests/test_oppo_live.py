"""The real OPlus endpoint and the real catalogue, opt-in, ~200 KB of traffic and no credential.

    UADCLAW_LIVE_FIRMWARE_TESTS=1 uv run pytest -n0 tests/test_oppo_live.py

Excluded from the default suite for the reason `test_samsung_live.py` gives: the default suite
is hermetic and runs in CI, and a reverse-engineered endpoint somebody else owns must not turn
their bad minute into a red build here. The catalogue half has a second reason — that site
escalates scraping to an IP-level block of its whole domain, so this file reads it exactly once
per run and nothing else in the repo reads it at all.

What only the live servers can prove is the crypto. `test_oppo_driver.py` mocks the endpoint
against a keypair it generated, which pins everything except the two constants that keypair
replaces: the region's real public key, and that what gets wrapped is the session key's base64
TEXT rather than its 32 raw bytes. Wrap the wrong one and this file's first test is the only
thing in the repo that notices.
"""

import os
import secrets
import time

import httpx
import pytest

from uadclaw.drivers.oppo import (
    GATE_USER_ID,
    REGIONS,
    OppoDriver,
    build_update_request,
    decrypt_update_response,
    download_host_allowed,
    resolved_package,
    synthesize_ota_version,
)
from uadclaw.firmware import select_ref
from uadclaw.settings import Settings

pytestmark = [
    pytest.mark.timeout(180),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_LIVE_FIRMWARE_TESTS") != "1",
        reason="opt-in: set UADCLAW_LIVE_FIRMWARE_TESTS=1 (talks to OPPO's servers)",
    ),
]

# End of life since 2023 and therefore stable: the same three values came back on 2026-08-12
# from two independent client implementations hours apart. A red here is the endpoint moving,
# which is the only thing this file is for.
CONTROL_MODEL = "RMX3706"
CONTROL_BUILD = "RMX3706_11.A.48_0480_202311201335"
CONTROL_SIZE = 7549973765
CONTROL_MD5 = "e027328b9c1f81382ec4d48b9e802914"
# One of the five modern export models that answer 2004 to everything.
REFUSED_MODEL = "CPH2653"
ZIP_MAGIC = b"PK\x03\x04"


def live_settings() -> Settings:
    return Settings(
        postgres_password="live-test-password",
        auth_password="live-test-admin-password",
        session_secret="live-test-session-secret",
    )


async def ask(client: httpx.AsyncClient, *, model: str, region_name: str, branch: str, major: str):
    region = REGIONS[region_name]
    key = secrets.token_bytes(32)
    wire, headers = build_update_request(
        model=model,
        ota_version=synthesize_ota_version(model, branch),
        region=region,
        android_major=major,
        key=key,
        iv=secrets.token_bytes(16),
        now_ms=int(time.time() * 1000),
    )
    response = await client.post(region.endpoint, content=wire, headers=headers)
    assert response.status_code == 200
    return decrypt_update_response(response.text, key)


async def test_a_synthesized_ota_version_still_resolves_the_control_package():
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
        code, document = await ask(
            client, model=CONTROL_MODEL, region_name="CN", branch="A", major="13"
        )

        assert code == 200
        package = resolved_package(document, model=CONTROL_MODEL)
        assert (package.ota_version, package.size, package.md5) == (
            CONTROL_BUILD,
            CONTROL_SIZE,
            CONTROL_MD5,
        )
        assert download_host_allowed(package.url)


async def test_the_export_model_refusal_has_not_moved():
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
        code, document = await ask(
            client, model=REFUSED_MODEL, region_name="GL", branch="F", major="16"
        )

    # 2004 with no body. If this ever answers 200 the hole closed and the catalogue stops being
    # load-bearing for this model; if a DIFFERENT model starts answering it, the driver's own
    # warning is what says so.
    assert (code, document) == (2004, None)


async def test_the_catalogue_lists_the_refused_model_and_its_gate_serves_zip_bytes():
    driver = OppoDriver(live_settings())

    refs = await driver.list_available()

    assert len(refs) > 50
    ref = select_ref(refs, device=REFUSED_MODEL)
    assert ref.md5 is not None
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0), follow_redirects=True) as client:
        # 64 KB rather than 8 GB: this proves the gate, its anti-leech header, the redirect to
        # the CDN and that what is behind it is an archive. Only the digest is left unproven,
        # and no ranged read can prove that one.
        response = await client.get(
            ref.url, headers={"userId": GATE_USER_ID, "Range": "bytes=0-65535"}
        )

    assert response.status_code == 206
    assert response.content[:4] == ZIP_MAGIC
