"""The real Samsung archive through the real driver, opt-in.

    UADCLAW_HEAVY_TESTS=1 \\
      UADCLAW_HEAVY_SAMSUNG_ENC4=/path/to/SM-S911U_2_..._fac.zip.enc4 \\
      UADCLAW_HEAVY_WORKDIR=/mnt/ssd-1/scratch/uadclaw-heavy \\
      uv run pytest -n0 -m heavy tests/test_samsung_fetch_heavy.py

Separate from `test_driver_chain_heavy.py`, which drives one device per OEM all the way to
merged facts, because Samsung **cannot get that far today** and the difference is the point:

    zip -> AP_*.tar.md5 (tar) -> *.img.lz4 (LZ4 frame) -> super.img -> lpunpack -> partitions

`uadclaw.unpack` dispatches on magic and has no handler for tar or for an LZ4 frame, so the
chain stops at the first step with `UnsupportedContainerError`. That is asserted below rather
than described, so the day `unpack` learns those two formats this file goes RED and is the
thing that tells the next task its seam moved.

What IS proven here is everything `acquire` owns: the whole FUS handshake, an 11.5 GB
streamed download, an in-place AES-128-ECB decrypt of every one of those bytes, the PKCS#7
strip, and a plaintext that Python's own zip reader opens and walks. Only the transport is
replaced — the local `.enc4` is streamed back as the response body, so the driver's own
download loop, decryptor and rename all run over multi-GB bytes rather than over a buffer no
production path ever produces.

Every figure is measured (2026-08-11, `SM-S911U`/`XAA`, build `S911USQS8FZG1`) and asserted
exactly. A decrypt that silently produced garbage yields a different digest, never an
exception.
"""

import os
import shutil
import zipfile
from pathlib import Path

import httpx
import pytest

from uadclaw.drivers.samsung import SamsungDriver
from uadclaw.firmware import FirmwareRef
from uadclaw.settings import Settings
from uadclaw.unpack import UnsupportedContainerError, unpack_to_partitions

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.timeout(5400),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_HEAVY_TESTS") != "1",
        reason="opt-in: set UADCLAW_HEAVY_TESTS=1 (needs the real 11.5 GB .enc4 and ~24 GB free)",
    ),
]

MODEL = "SM-S911U"
REGION = "XAA"
BUILD = "S911USQS8FZG1_XAA"

# The encrypted archive FUS serves, and the plaintext it decrypts to. The two differ by the
# 14 bytes of PKCS#7 padding stripped off the end.
ENCRYPTED_BYTES = 11_565_187_312
PLAINTEXT_BYTES = 11_565_187_298
PLAINTEXT_SHA256 = "fc563d5b8bff839eaacdf0f9d5674ee043309fb5153d9b73542251d1ae6c85ac"
# What FUS declared for the ENCRYPTED body, and what 11,565,187,312 real bytes hash to. The
# plaintext's CRC32 is 3102581189 and matches nothing, which is how the field's meaning was
# established rather than guessed.
ENCRYPTED_CRC32 = 949352961

# The six members of the decrypted zip, with the uncompressed size of each. Deflated, and the
# whole archive is ZIP64: the AP member alone is past the 4 GB 32-bit field.
MEMBERS = {
    "BL_S911USQS8FZG1_S911USQS8FZG1_MQB111837160_REV00_user_low_ship_MULTI_CERT.tar.md5": 123177073,
    "AP_S911USQS8FZG1_S911USQS8FZG1_MQB111837160_REV00_user_low_ship_MULTI_CERT_meta_OS16"
    ".tar.md5": 11465093243,
    "CP_S911USQS8FZG1_CP35310094_MQB111837160_REV00_user_low_ship_MULTI_CERT.tar.md5": 76656750,
    "HOME_CSC_OYN_S911UOYN8FZG1_MQB111837160_REV00_user_low_ship_MULTI_CERT.tar.md5": 183296109,
    "CSC_OYN_S911UOYN8FZG1_MQB111837160_REV00_user_low_ship_MULTI_CERT.tar.md5": 183326824,
    "USERDATA_VZW_S911USQS8FZG1_S911USQS8FZG1_MQB111837160_REV00_user_low_ship_MULTI_CERT"
    ".tar.md5": 1847439483,
}

REQUIRED_FREE_BYTES = 24 * 1024**3


def _settings() -> Settings:
    return Settings(
        postgres_password="test-only-password",
        auth_password="test-only-admin-password",
        session_secret="test-only-session-secret",
        samsung_models=MODEL,
        samsung_regions=REGION,
    )


def _local_enc4() -> Path:
    value = os.environ.get("UADCLAW_HEAVY_SAMSUNG_ENC4", "")
    if not value:
        pytest.skip("set UADCLAW_HEAVY_SAMSUNG_ENC4 to the downloaded .enc4 for this build")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"UADCLAW_HEAVY_SAMSUNG_ENC4 points at {path}, which is not on this box")
    if path.stat().st_size != ENCRYPTED_BYTES:
        pytest.skip(
            f"{path} is {path.stat().st_size} bytes, not the {ENCRYPTED_BYTES} this build "
            "publishes; Samsung has moved on and the pinned digests below describe the old one"
        )
    return path


def _workdir() -> Path:
    root = Path(os.environ.get("UADCLAW_HEAVY_WORKDIR", "/var/tmp/uadclaw-heavy"))
    work = root / "samsung"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < REQUIRED_FREE_BYTES:
        pytest.skip(f"{work} has {free / 1024**3:.1f} GB free, need ~24 GB")
    return work


def _client(enc4: Path) -> httpx.AsyncClient:
    """The index and the FUS handshake off the committed fixtures; the binary off local disk,
    STREAMED, so the driver's own chunking and in-place decrypt run over the real bytes."""
    nonces = iter([f"nonce-{index:04d}" for index in range(3)])

    async def body():
        with enc4.open("rb") as fh:
            while chunk := fh.read(8 * 1024 * 1024):
                yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "version.xml" in url:
            return httpx.Response(
                200, text=(FIXTURES / "samsung_version_index.xml").read_text(encoding="utf-8")
            )
        if "BinaryForMass" in url:
            return httpx.Response(200, content=body())
        headers = {"NONCE": next(nonces)}
        if url.endswith("BinaryInform.do"):
            return httpx.Response(
                200,
                text=(FIXTURES / "samsung_binary_inform.xml").read_text(encoding="utf-8"),
                headers=headers,
            )
        return httpx.Response(
            200,
            text="<FUSMsg><FUSBody><Results><Status>S00</Status></Results></FUSBody></FUSMsg>",
            headers=headers,
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_the_real_archive_decrypts_to_a_zip_the_stdlib_walks(tmp_path):
    enc4 = _local_enc4()
    work = _workdir()
    driver = SamsungDriver(_settings(), client=_client(enc4))
    ref = FirmwareRef(
        driver="samsung",
        device=MODEL,
        build=BUILD,
        url=f"https://fota-cloud-dn.ospserver.net/firmware/{REGION}/{MODEL}/version.xml",
    )

    # The committed fixture is what the driver derives the key and the CRC from, so it has to
    # still describe the build these numbers were measured on.
    inform = (FIXTURES / "samsung_binary_inform.xml").read_text(encoding="utf-8")
    assert f"<BINARY_CRC><Data>{ENCRYPTED_CRC32}</Data>" in inform
    assert f"<BINARY_BYTE_SIZE><Data>{ENCRYPTED_BYTES}</Data>" in inform

    archive = await driver.fetch(ref, work)

    assert archive.path.name == "samsung-SM-S911U-S911USQS8FZG1_XAA.zip"
    # 14 bytes shorter than the download: that is the PKCS#7 padding, stripped.
    assert archive.path.stat().st_size == PLAINTEXT_BYTES
    assert ENCRYPTED_BYTES - PLAINTEXT_BYTES == 14
    assert archive.sha256 == PLAINTEXT_SHA256
    # Verified against the CRC32 FUS published for the encrypted body, checked over every one
    # of the 11,565,187,312 bytes during the decrypt pass.
    assert archive.integrity_verified is True
    # Nothing encrypted survives: the decrypt is in place and the file is renamed, so a `.enc4`
    # left here would be a second copy of 11.5 GB that retention never hears about.
    assert [path.name for path in work.iterdir()] == [archive.path.name]

    with archive.path.open("rb") as fh:
        assert fh.read(4) == b"PK\x03\x04"
    with zipfile.ZipFile(archive.path) as zf:
        assert {info.filename: info.file_size for info in zf.infolist()} == MEMBERS
        # Reads and CRCs every member: 11.5 GB of deflate that only decrypts correctly if
        # every block of the in-place pass landed at the offset its ciphertext came from.
        assert zf.testzip() is None


async def test_the_unpack_chain_stops_at_the_tar_members_this_repo_cannot_open(tmp_path):
    """The task-11 verify line ("one full device end to end producing merged facts") is NOT
    met for Samsung, and this is where it stops. Delete this test when `unpack` learns tar and
    LZ4; until then it is the pin that keeps the gap from being reported as closed.
    """
    enc4 = _local_enc4()
    work = _workdir()
    driver = SamsungDriver(_settings(), client=_client(enc4))
    ref = FirmwareRef(
        driver="samsung",
        device=MODEL,
        build=BUILD,
        url=f"https://fota-cloud-dn.ospserver.net/firmware/{REGION}/{MODEL}/version.xml",
    )
    archive = await driver.fetch(ref, work)

    with pytest.raises(UnsupportedContainerError) as excinfo:
        await unpack_to_partitions(archive.path, work / "unpack", settings=_settings())

    assert "no readable .img member" in str(excinfo.value)
    assert all(name.endswith(".tar.md5") for name in MEMBERS)
