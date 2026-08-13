"""The real Samsung archive through the real driver, opt-in.

    UADCLAW_HEAVY_TESTS=1 \\
      UADCLAW_HEAVY_SAMSUNG_ENC4=/path/to/SM-S911U_2_..._fac.zip.enc4 \\
      UADCLAW_HEAVY_WORKDIR=/mnt/ssd-1/scratch/uadclaw-heavy \\
      uv run pytest -n0 -m heavy tests/test_samsung_fetch_heavy.py

Separate from `test_driver_chain_heavy.py`, which drives one device per OEM against a real
`uad_lists.json`, because Samsung's chain is two steps deeper than any of those and that
difference is what this file is for:

    zip -> AP_*.tar.md5 (tar) -> *.img.lz4 (LZ4 frame) -> super.img -> lpunpack -> partitions

Proven here: everything `acquire` owns — the whole FUS handshake, an 11.5 GB streamed
download, an in-place AES-128-ECB decrypt of every one of those bytes, the PKCS#7 strip — and
then the whole way down to merged facts in Postgres. Only the transport is replaced: the
local `.enc4` is streamed back as the response body, so the driver's own download loop,
decryptor and rename all run over multi-GB bytes rather than over a buffer no production path
ever produces.

Every figure is measured (`SM-S911U`/`XAA`, build `S911USQS8FZG1`: the archive 2026-08-11,
the chain 2026-08-12) and asserted exactly. Neither half fails loudly on its own — a decrypt
that produced garbage yields a different digest rather than an exception, and a chain that
quietly stops extracting yields a smaller count rather than one.
"""

import os
import shutil
import zipfile
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from uadclaw import firmware as firmware_module
from uadclaw import jobs as jobs_module
from uadclaw.drivers.samsung import SamsungDriver
from uadclaw.firmware import FirmwareRef
from uadclaw.models import JobKind, PackageFact, PackageObservation
from uadclaw.settings import Settings, get_settings
from uadclaw.stages import (
    ARTIFACTS_DIRNAME,
    acquire_stage,
    extract_facts_stage,
    read_state,
    unpack_stage,
)
from uadclaw.worker import StageContext

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.timeout(5400),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_HEAVY_TESTS") != "1",
        reason="opt-in: set UADCLAW_HEAVY_TESTS=1 (needs the real 11.5 GB .enc4 and ~44 GB free)",
    ),
]

MODEL = "SM-S911U"
REGION = "XAA"
BUILD = "S911USQS8FZG1_XAA"

# What the whole chain yields, measured 2026-08-12 and asserted exactly. Every partition the
# super carries is EROFS on this build, which is why `fsck.erofs` and not 7z is on the path,
# and the last two come out of `HOME_CSC_` rather than the super at all — `CSC_` ships the
# same two images and they are dropped as byte-identical duplicates.
PARTITIONS = [
    "odm",
    "product",
    "system",
    "system_dlkm",
    "system_ext",
    "vendor",
    "vendor_dlkm",
    "prism",
    "optics",
]
# Five of the nine yield no APK: `odm` is a 10-file stub here, `prism` and `optics` hold 397
# and 72 entries of CSC data, and the two `_dlkm` partitions are kernel modules.
APKS_BY_PARTITION = {"product": 79, "system": 407, "system_ext": 13, "vendor": 10}
CHAIN = {"apks": 509, "artifacts": 1143, "packages": 491, "parse_failures": 0}

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

# Measured 2026-08-12 on the run below, `du -sb` every 3 s: this scratch tree peaks at
# 35,422,204,778 bytes, when the 11.57 GB decrypted archive, the LZ4-decoded `super.img`
# (11.37 GB, still sparse) and the raw image `simg2img` writes from it (12.66 GB) are all on
# disk at once. Deepest-peaking chain in the project by some way, and the reason the old 24 GB
# figure — sized for the download and the decrypt alone — could not have run it.
REQUIRED_FREE_BYTES = 44 * 1024**3

TOOLCHAIN = ("7z", "fsck.erofs", "simg2img", "lpunpack")


def _require_toolchain() -> None:
    missing = [tool for tool in TOOLCHAIN if shutil.which(tool) is None]
    if missing:
        pytest.skip(f"needs the unpacking toolchain on PATH: {', '.join(missing)}")


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


async def test_samsung_sm_s911u_end_to_end(db_env, db_session_factory, monkeypatch, tmp_path):
    """The task-11 verify line for Samsung: one full device to merged facts, through the same
    stage handlers the worker runs, with only the transport replaced.

    `filter` and `rule_ladder` are deliberately not here — they need a real `uad_lists.json`,
    which `test_driver_chain_heavy.py` owns for the three OEMs that have one on this box.
    What this pins is everything Samsung alone reaches: the FUS handshake, the decrypt, and
    the two container steps no other OEM takes.
    """
    _require_toolchain()
    enc4 = _local_enc4()
    work = _workdir()
    monkeypatch.setenv("SAMSUNG_MODELS", MODEL)
    monkeypatch.setenv("SAMSUNG_REGIONS", REGION)
    get_settings.cache_clear()
    monkeypatch.setattr(
        firmware_module,
        "_driver_factories",
        lambda: {"samsung": lambda settings: SamsungDriver(settings, client=_client(enc4))},
    )

    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session,
            kind=JobKind.FIRMWARE_ANALYSIS.value,
            params={"driver": "samsung", "device": MODEL, "build": BUILD},
        )
        job_id = job.id
    ctx = StageContext(
        job_id=job_id, attempt=1, scratch_dir=work, session_factory=db_session_factory
    )

    await acquire_stage(ctx)
    await unpack_stage(ctx)
    await extract_facts_stage(ctx)

    state = read_state(work)
    assert state.integrity_verified is True
    assert state.archive_sha256 == PLAINTEXT_SHA256
    assert state.partitions == PARTITIONS
    assert {
        "apks": state.apk_count,
        "artifacts": state.artifact_count,
        "packages": state.package_count,
        "parse_failures": state.parse_failed_count,
    } == CHAIN
    async with db_session_factory() as session:
        rows = (
            await session.execute(
                select(PackageObservation.partition, func.count()).group_by(
                    PackageObservation.partition
                )
            )
        ).all()
        merged = (await session.execute(select(func.count()).select_from(PackageFact))).scalar()
    assert {partition: count for partition, count in rows} == APKS_BY_PARTITION
    assert merged == CHAIN["packages"]
    # Facts-only retention: 23 GB of archive, super and partition images, all gone.
    assert state.archive_path is None
    assert not list((work / "unpack").rglob("*.img"))
    assert not list((work / ARTIFACTS_DIRNAME).rglob("*.apk"))
