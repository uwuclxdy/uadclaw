"""The real Samsung servers, opt-in, ~70 KB of traffic and no credential.

    UADCLAW_LIVE_FIRMWARE_TESTS=1 uv run pytest -n0 tests/test_samsung_live.py

Excluded from the default suite, mirroring `UADCLAW_LIVE_API_TESTS=1` and `UADCLAW_HEAVY_TESTS=1`.
Not because it is slow or billed — it is neither — but because the default suite is hermetic
and runs in CI: a test that talks to a reverse-engineered endpoint somebody else owns turns
somebody else's bad minute, or a blocked runner IP, into a red build on this repo's code.

What only the live server can prove is that the request shape this module ships is one FUS
still accepts. `test_firmware_drivers.py` pins every decision this code makes against captured
responses; a mock can only ever prove the code does what the mock's author expected, which is
the wrong question for a wire format nobody documented.

**This costs 64 KB and proves the whole chain.** Because the payload is AES-ECB, any 16-byte
aligned range decrypts on its own — so a ranged GET of the first block, decrypted, exercises
the nonce, the signature, the inform, the init call, the download authorisation AND the key
derivation. Every one of those fails closed here, at a page of traffic, rather than after an
11.5 GB transfer.
"""

import os

import httpx
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from uadclaw.drivers.samsung import AES_BLOCK_BYTES, SamsungDriver
from uadclaw.firmware import FirmwareRef, select_ref
from uadclaw.settings import Settings

pytestmark = [
    pytest.mark.timeout(180),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_LIVE_FIRMWARE_TESTS") != "1",
        reason="opt-in: set UADCLAW_LIVE_FIRMWARE_TESTS=1 (talks to Samsung's servers)",
    ),
]

MODEL = "SM-S911U"
REGION = "XAA"
ZIP_MAGIC = b"PK\x03\x04"


def live_settings(**overrides) -> Settings:
    return Settings(
        postgres_password="test-only-password",
        auth_password="test-only-admin-password",
        session_secret="test-only-session-secret",
        **{"samsung_models": MODEL, "samsung_regions": REGION, **overrides},
    )


async def test_the_live_index_still_publishes_the_shape_this_driver_parses():
    refs = await SamsungDriver(live_settings()).list_available()

    assert [ref.device for ref in refs] == [MODEL]
    assert refs[0].build.endswith(f"_{REGION}")
    assert refs[0].android_version is not None


async def test_a_pair_that_does_not_exist_still_answers_403_rather_than_an_empty_200():
    """The existence oracle the whole enumeration rests on. `SM-A546B` is published under EUX
    and 403s under DBT: if a nonexistent pair ever starts answering 200 with an empty body,
    enumeration silently returns a catalogue with holes instead of raising."""
    driver = SamsungDriver(live_settings(samsung_models="SM-A546B", samsung_regions="DBT,EUX"))

    refs = await driver.list_available()

    assert [ref.build.rsplit("_", 1)[1] for ref in refs] == ["EUX"]


async def test_the_live_handshake_resolves_a_binary_that_decrypts_to_the_zip_magic():
    """The entire auth chain, proven on 64 KB.

    Skip the init call and this same request answers HTTP 401 with a 119-byte JSON body, so a
    206 here is positive evidence that every step ran — not merely that nothing raised.
    """
    driver = SamsungDriver(live_settings())
    async with httpx.AsyncClient(timeout=httpx.Timeout(60), follow_redirects=True) as client:
        refs = await driver.list_available()
        ref: FirmwareRef = select_ref(refs, device=MODEL)

        session, binary = await driver.resolve(client, ref)

        assert binary.size % AES_BLOCK_BYTES == 0
        assert binary.filename.endswith(".zip.enc4")
        assert binary.region == REGION
        response = await client.get(
            binary.download_url, headers={**session.headers(), "Range": "bytes=0-65535"}
        )

    assert response.status_code == httpx.codes.PARTIAL_CONTENT, response.text[:200]
    assert len(response.content) == 65536
    decryptor = Cipher(algorithms.AES(binary.key), modes.ECB()).decryptor()
    plaintext = decryptor.update(response.content[:AES_BLOCK_BYTES]) + decryptor.finalize()
    # The hex goes in the message because a shell-mangled backslash in an expected-bytes
    # literal reads exactly like a wrong key, and the two want opposite responses.
    assert plaintext[:4] == ZIP_MAGIC, (
        f"first block decrypted to {plaintext.hex()}, expected it to open {ZIP_MAGIC.hex()}"
    )
