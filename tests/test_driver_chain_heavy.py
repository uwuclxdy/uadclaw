"""One real device per OEM, driven through the real pipeline stages, end to end and opt-in.

Task 11's verify line: "one full device end to end producing merged facts". So this drives
`acquire -> unpack -> extract_facts -> corpus_graph -> filter -> rule_ladder` — the same
handlers the worker runs — with the OEM's own driver on the path, resolved through the real
registry so `get_driver`'s enable/disable check is exercised too.

Only the transport is replaced. Each driver's `list_available` reads the committed index
fixture and its `fetch` runs verbatim: Xiaomi's md5 is checked against the digest its real
index published for this exact build, Nothing's three volumes are joined by the driver's
own join, and Samsung's `.enc4` is decrypted in place with the key the committed inform
fixture resolves, its encrypted body's CRC32 checked over every byte. What the mock does NOT
do is re-download 13 GB on every run.

    UADCLAW_HEAVY_TESTS=1 \\
      UADCLAW_HEAVY_XIAOMI_ZIP=/path/to/miui_WATERGlobal_V14.0.24.0.TGOMIXM_....zip \\
      UADCLAW_HEAVY_NOTHING_DIR=/path/holding/FroggerPro_B4.1-260723-1820-image-logical.7z.00N \\
      UADCLAW_HEAVY_MOTOROLA_ZIP=/path/to/RTWO_RETAIL_15_V1TRS35H.60-33-7....zip \\
      UADCLAW_HEAVY_SAMSUNG_ENC4=/path/to/SM-S911U_2_..._fac.zip.enc4 \\
      UADCLAW_HEAVY_UPSTREAM_LIST=/path/to/uad_lists.json \\
      UADCLAW_HEAVY_WORKDIR=/var/tmp/uadclaw-heavy \\
      uv run pytest -n0 -m heavy tests/test_driver_chain_heavy.py

Every count below was measured on 2026-08-11 and is asserted exactly. A chain that quietly
stops extracting returns a smaller number, never an exception: the Xiaomi build's three EROFS
partitions and the Motorola super's six are the whole point of pinning them per partition.
Samsung is the one exception to "asserted exactly": its chain figures were measured with
`test_samsung_fetch_heavy.py` on 2026-08-12, but `filter` and `rule_ladder` have never run for
a Samsung build, so `SAMSUNG_QUEUE`'s two upstream columns are placeholders that red on
purpose until a heavy run measures them and pins the columns.
"""

import json
import os
import shutil
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from uadclaw import firmware as firmware_module
from uadclaw import jobs as jobs_module
from uadclaw.drivers.motorola import MotorolaDriver
from uadclaw.drivers.nothing import NothingDriver
from uadclaw.drivers.samsung import SamsungDriver
from uadclaw.drivers.xiaomi import XiaomiDriver
from uadclaw.models import JobKind, PackageAnalysis, PackageFact, PackageObservation
from uadclaw.settings import get_settings
from uadclaw.stages import (
    ARTIFACTS_DIRNAME,
    acquire_stage,
    corpus_graph_stage,
    extract_facts_stage,
    filter_stage,
    read_state,
    rule_ladder_stage,
    unpack_stage,
)
from uadclaw.worker import StageContext

FIXTURES = Path(__file__).parent / "fixtures"
UPSTREAM_LIST = os.environ.get("UADCLAW_HEAVY_UPSTREAM_LIST", "")

# Peak on the Motorola run: the 5.8 GB zip, its 6.4 GB of sparse chunks, the ~7 GB raw super
# lpunpack reads, and the partitions it writes. The other two peak lower.
REQUIRED_FREE_BYTES = 40 * 1024**3

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.timeout(5400),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_HEAVY_TESTS") != "1",
        reason="opt-in: set UADCLAW_HEAVY_TESTS=1 (needs ~44 GB scratch and the toolchain)",
    ),
]

TOOLCHAIN = ("7z", "fsck.erofs", "simg2img", "lpunpack", "payload-dumper-go")


def _require_toolchain() -> None:
    missing = [tool for tool in TOOLCHAIN if shutil.which(tool) is None]
    if missing:
        pytest.skip(f"needs the unpacking toolchain on PATH: {', '.join(missing)}")


def _require_workdir(name: str, required_bytes: int = REQUIRED_FREE_BYTES) -> Path:
    root = Path(os.environ.get("UADCLAW_HEAVY_WORKDIR", "/var/tmp/uadclaw-heavy"))
    work = root / name
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < required_bytes:
        pytest.skip(
            f"{work} has {free / 1024**3:.1f} GB free, need ~{required_bytes / 1024**3:.0f} GB"
        )
    return work


def _local_source(env_var: str) -> Path:
    value = os.environ.get(env_var, "")
    if not value:
        pytest.skip(f"set {env_var} to the downloaded firmware for this OEM")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{env_var} points at {path}, which is not on this box")
    return path


def _stream(path: Path) -> httpx.Response:
    """Serve a local firmware file as a streamed body, so the driver's own download loop —
    chunking, hashing, the `.part` rename — runs over multi-GB bytes rather than being
    handed a buffer no production path ever produces."""

    async def body():
        with path.open("rb") as fh:
            while chunk := fh.read(8 * 1024 * 1024):
                yield chunk

    return httpx.Response(200, content=body())


def _upstream_list() -> str:
    if not UPSTREAM_LIST or not Path(UPSTREAM_LIST).is_file():
        pytest.skip("set UADCLAW_HEAVY_UPSTREAM_LIST to a real uad_lists.json")
    return UPSTREAM_LIST


async def _run_chain(ctx: StageContext) -> dict[str, int]:
    """Every stage the worker would run, in order, with no stubs between them."""
    await acquire_stage(ctx)
    await unpack_stage(ctx)
    await extract_facts_stage(ctx)
    await corpus_graph_stage(ctx)
    await filter_stage(ctx)
    await rule_ladder_stage(ctx)
    state = read_state(ctx.scratch_dir)
    return {
        "apks": state.apk_count,
        "artifacts": state.artifact_count,
        "packages": state.package_count,
        "parse_failures": state.parse_failed_count,
    }


async def _queue_counts(session_factory) -> dict[str, int]:
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(PackageAnalysis.filter_verdict, func.count()).group_by(
                    PackageAnalysis.filter_verdict
                )
            )
        ).all()
        merged = (await session.execute(select(func.count()).select_from(PackageFact))).scalar()
    counts = {str(verdict): count for verdict, count in rows}
    counts["merged_packages"] = int(merged or 0)
    return counts


async def _make_job(session_factory, params: dict[str, str]):
    async with session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.FIRMWARE_ANALYSIS.value, params=params
        )
        return job.id


def _register(monkeypatch, name: str, factory) -> None:
    monkeypatch.setattr(firmware_module, "_driver_factories", lambda: {name: factory})


async def _apk_partitions(session_factory) -> dict[str, int]:
    """One observation row per APK, so this counts what actually reached Postgres rather than
    what is on disk — by the time the chain ends, retention has deleted every APK, and
    counting files there returned `{}` for a run that had stored 227 of them."""
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(PackageObservation.partition, func.count()).group_by(
                    PackageObservation.partition
                )
            )
        ).all()
    return {partition: count for partition, count in rows}


# --- Xiaomi: zip -> payload.bin -> payload-dumper-go -> three EROFS partitions --------------

XIAOMI_DEVICE = "water_global"
XIAOMI_BUILD = "V14.0.24.0.TGOMIXM"
XIAOMI_APKS_BY_PARTITION = {"product": 79, "system": 136, "vendor": 12}
XIAOMI_CHAIN = {"apks": 227, "artifacts": 324, "packages": 204, "parse_failures": 0}
XIAOMI_QUEUE = {"already_upstream": 182, "queued": 22, "merged_packages": 204}


async def test_xiaomi_water_global_end_to_end(db_env, db_session_factory, monkeypatch, tmp_path):
    _require_toolchain()
    archive = _local_source("UADCLAW_HEAVY_XIAOMI_ZIP")
    work = _require_workdir("xiaomi")
    monkeypatch.setenv("UPSTREAM_LIST_PATH", _upstream_list())
    get_settings.cache_clear()

    index = (FIXTURES / "xiaomi_latest_index.yml").read_text(encoding="utf-8")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("latest.yml"):
            return httpx.Response(200, text=index)
        return _stream(archive)

    _register(
        monkeypatch,
        "xiaomi",
        lambda settings: XiaomiDriver(
            settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    job_id = await _make_job(
        db_session_factory,
        {"driver": "xiaomi", "device": XIAOMI_DEVICE, "build": XIAOMI_BUILD},
    )
    ctx = StageContext(
        job_id=job_id, attempt=1, scratch_dir=work, session_factory=db_session_factory
    )

    counts = await _run_chain(ctx)

    assert counts == XIAOMI_CHAIN
    state = read_state(work)
    # The md5 the tracker published for this build, checked against 1.5 GB of real bytes.
    assert state.integrity_verified is True
    assert state.partitions == ["product", "system", "vendor"]
    assert await _apk_partitions(db_session_factory) == XIAOMI_APKS_BY_PARTITION
    # The index's `bigota` link is rewritten before anything is requested.
    assert seen[-1].startswith("https://cdnorg.d.miui.com/")
    assert await _queue_counts(db_session_factory) == XIAOMI_QUEUE
    # Facts-only retention: no archive, no partition image, no APK left on disk.
    assert state.archive_path is None
    assert not list((work / "unpack").rglob("*.img"))
    assert not list((work / ARTIFACTS_DIRNAME).rglob("*.apk"))


# --- Nothing: three 7z volumes -> one archive -> seven partition images ---------------------

NOTHING_DEVICE = "FroggerPro"
NOTHING_BUILD = "B4.1-260723-1820"
# sha256 of the three volumes concatenated in order, i.e. `cat *.7z.00* | sha256sum` over
# the 6,139,281,340 bytes the release publishes. Pins the join as byte-exact rather than
# merely producing something 7z happens to open.
NOTHING_ARCHIVE_SHA256 = "b6c8528eab7facbe21aa3adad9e7f892025d9975cc4dc4bb57673f8da3720c2a"
NOTHING_APKS_BY_PARTITION = {"odm": 4, "product": 186, "system": 83, "system_ext": 66, "vendor": 34}
NOTHING_CHAIN = {"apks": 374, "artifacts": 668, "packages": 353, "parse_failures": 1}
NOTHING_QUEUE = {"already_upstream": 255, "queued": 98, "merged_packages": 353}


async def test_nothing_frogger_pro_end_to_end(db_env, db_session_factory, monkeypatch, tmp_path):
    _require_toolchain()
    volumes_dir = _local_source("UADCLAW_HEAVY_NOTHING_DIR")
    work = _require_workdir("nothing")
    monkeypatch.setenv("UPSTREAM_LIST_PATH", _upstream_list())
    get_settings.cache_clear()

    releases = json.loads((FIXTURES / "nothing_releases.json").read_text(encoding="utf-8"))
    tag = f"{NOTHING_DEVICE}_{NOTHING_BUILD}"
    release = next(item for item in releases if item["tag_name"] == tag)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(f"/tags/{tag}"):
            return httpx.Response(200, text=json.dumps(release))
        if request.url.path.endswith("/releases"):
            return httpx.Response(200, text=json.dumps(releases))
        volume = volumes_dir / Path(request.url.path).name
        if not volume.is_file():
            return httpx.Response(404, content=b"no such volume")
        return _stream(volume)

    _register(
        monkeypatch,
        "nothing",
        lambda settings: NothingDriver(
            settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    job_id = await _make_job(
        db_session_factory,
        {"driver": "nothing", "device": NOTHING_DEVICE, "build": NOTHING_BUILD},
    )
    ctx = StageContext(
        job_id=job_id, attempt=1, scratch_dir=work, session_factory=db_session_factory
    )

    counts = await _run_chain(ctx)

    assert counts == NOTHING_CHAIN
    state = read_state(work)
    # The release's published hashes cover the images inside the archive, not the download.
    assert state.integrity_verified is False
    assert state.archive_sha256 == NOTHING_ARCHIVE_SHA256
    assert await _apk_partitions(db_session_factory) == NOTHING_APKS_BY_PARTITION
    assert await _queue_counts(db_session_factory) == NOTHING_QUEUE
    assert state.archive_path is None
    assert not list((work / "unpack").rglob("*.img"))
    assert not list((work / ARTIFACTS_DIRNAME).rglob("*.apk"))


# --- Motorola: zip -> 14 sparse chunks -> simg2img -> super.img -> lpunpack -----------------

MOTOROLA_DEVICE = "rtwo"
MOTOROLA_BUILD = "V1TRS35H.60-33-7_RETAIL"
MOTOROLA_APKS_BY_PARTITION = {"product": 265, "system": 83, "system_ext": 82, "vendor": 16}
MOTOROLA_CHAIN = {"apks": 446, "artifacts": 993, "packages": 427, "parse_failures": 0}
MOTOROLA_QUEUE = {
    "already_upstream": 349,
    "auto_generated_rro": 2,
    "queued": 76,
    "merged_packages": 427,
}


async def test_motorola_rtwo_end_to_end(db_env, db_session_factory, monkeypatch, tmp_path):
    _require_toolchain()
    archive = _local_source("UADCLAW_HEAVY_MOTOROLA_ZIP")
    work = _require_workdir("motorola")
    monkeypatch.setenv("UPSTREAM_LIST_PATH", _upstream_list())
    get_settings.cache_clear()

    listings = json.loads((FIXTURES / "motorola_h5ai_listings.json").read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("index.php"):
            path = httpx.QueryParams(request.content.decode()).get("items[href]", "")
            return httpx.Response(200, json=listings.get(path, {"items": []}))
        return _stream(archive)

    _register(
        monkeypatch,
        "motorola",
        lambda settings: MotorolaDriver(
            settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    monkeypatch.setenv("MOTOROLA_DEVICES", MOTOROLA_DEVICE)
    get_settings.cache_clear()
    job_id = await _make_job(
        db_session_factory,
        {"driver": "motorola", "device": MOTOROLA_DEVICE, "build": MOTOROLA_BUILD},
    )
    ctx = StageContext(
        job_id=job_id, attempt=1, scratch_dir=work, session_factory=db_session_factory
    )

    counts = await _run_chain(ctx)

    assert counts == MOTOROLA_CHAIN
    state = read_state(work)
    # The h5ai listing publishes no checksum of any kind.
    assert state.integrity_verified is False
    assert await _apk_partitions(db_session_factory) == MOTOROLA_APKS_BY_PARTITION
    assert await _queue_counts(db_session_factory) == MOTOROLA_QUEUE
    assert state.archive_path is None
    assert not list((work / "unpack").rglob("*.img"))
    assert not list((work / ARTIFACTS_DIRNAME).rglob("*.apk"))


# --- Samsung: zip -> tar -> LZ4 frame -> sparse -> super -> EROFS --------------------------

SAMSUNG_DEVICE = "SM-S911U"
SAMSUNG_REGION = "XAA"
SAMSUNG_BUILD = "S911USQS8FZG1_XAA"
# sha256 of the decrypted archive. FUS publishes no digest of either form, so
# `integrity_verified` is True because the CRC32 it declared for the ENCRYPTED body
# (949352961) was checked over every one of the 11,565,187,312 bytes during the in-place
# decrypt. The archive is 14 bytes shorter than the encrypted body: PKCS#7 padding, stripped.
SAMSUNG_ENCRYPTED_BYTES = 11_565_187_312
SAMSUNG_ENCRYPTED_CRC32 = 949352961
SAMSUNG_ARCHIVE_SHA256 = "fc563d5b8bff839eaacdf0f9d5674ee043309fb5153d9b73542251d1ae6c85ac"
SAMSUNG_APKS_BY_PARTITION = {"product": 79, "system": 407, "system_ext": 13, "vendor": 10}
SAMSUNG_CHAIN = {"apks": 509, "artifacts": 1143, "packages": 491, "parse_failures": 0}
SAMSUNG_QUEUE = {
    # MEASURED: pinned after the first heavy run; this placeholder reds on purpose, and the
    # failure diff carries the real dict — copy its KEY SET too, since `group_by` omits
    # zero-count groups and a Samsung run may add an `auto_generated_rro` key.
    "already_upstream": 0,
    "queued": 0,
    "merged_packages": 491,
}
# Measured 2026-08-12 on the real chain: the decrypted archive (11.57 GB), the LZ4-decoded
# `super.img` (11.37 GB, still sparse) and the raw image simg2img writes from it (12.66 GB)
# are all on disk at once, a peak of 35.4 GB — deeper than the other three rows' 40 GB budget.
SAMSUNG_REQUIRED_FREE_BYTES = 44 * 1024**3


def _local_enc4() -> Path:
    value = os.environ.get("UADCLAW_HEAVY_SAMSUNG_ENC4", "")
    if not value:
        pytest.skip("set UADCLAW_HEAVY_SAMSUNG_ENC4 to the downloaded .enc4 for this build")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"UADCLAW_HEAVY_SAMSUNG_ENC4 points at {path}, which is not on this box")
    if path.stat().st_size != SAMSUNG_ENCRYPTED_BYTES:
        pytest.skip(
            f"{path} is {path.stat().st_size} bytes, not the {SAMSUNG_ENCRYPTED_BYTES} this "
            "build publishes; Samsung has moved on and the pinned digests below describe the "
            "old one"
        )
    return path


async def test_samsung_sm_s911u_end_to_end_with_upstream_figures(
    db_env, db_session_factory, monkeypatch, tmp_path
):
    """Task-14's verify line for Samsung: `filter` and `rule_ladder` over a real
    `uad_lists.json`, filling the two upstream columns `test_samsung_fetch_heavy.py` could not
    measure. The chain figures are asserted beside them because they are the proof that these
    queue counts describe the same build that run pinned.
    """
    _require_toolchain()
    enc4 = _local_enc4()
    work = _require_workdir("samsung", required_bytes=SAMSUNG_REQUIRED_FREE_BYTES)
    monkeypatch.setenv("UPSTREAM_LIST_PATH", _upstream_list())
    monkeypatch.setenv("SAMSUNG_MODELS", SAMSUNG_DEVICE)
    monkeypatch.setenv("SAMSUNG_REGIONS", SAMSUNG_REGION)
    get_settings.cache_clear()

    # The committed fixture is what the driver derives the key and the CRC from, so it has to
    # still describe the build these numbers were measured on.
    inform = (FIXTURES / "samsung_binary_inform.xml").read_text(encoding="utf-8")
    assert f"<BINARY_CRC><Data>{SAMSUNG_ENCRYPTED_CRC32}</Data>" in inform
    assert f"<BINARY_BYTE_SIZE><Data>{SAMSUNG_ENCRYPTED_BYTES}</Data>" in inform

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
            return httpx.Response(200, text=inform, headers=headers)
        return httpx.Response(
            200,
            text="<FUSMsg><FUSBody><Results><Status>S00</Status></Results></FUSBody></FUSMsg>",
            headers=headers,
        )

    _register(
        monkeypatch,
        "samsung",
        lambda settings: SamsungDriver(
            settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    job_id = await _make_job(
        db_session_factory,
        {"driver": "samsung", "device": SAMSUNG_DEVICE, "build": SAMSUNG_BUILD},
    )
    ctx = StageContext(
        job_id=job_id, attempt=1, scratch_dir=work, session_factory=db_session_factory
    )

    counts = await _run_chain(ctx)

    assert counts == SAMSUNG_CHAIN
    state = read_state(work)
    assert state.integrity_verified is True
    assert state.archive_sha256 == SAMSUNG_ARCHIVE_SHA256
    assert await _apk_partitions(db_session_factory) == SAMSUNG_APKS_BY_PARTITION
    assert await _queue_counts(db_session_factory) == SAMSUNG_QUEUE
    assert state.archive_path is None
    assert not list((work / "unpack").rglob("*.img"))
    assert not list((work / ARTIFACTS_DIRNAME).rglob("*.apk"))
