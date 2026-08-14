"""Fact extraction against two real Android corpora, end to end and opt-in.

`test_facts.py` proves the merge with synthetic facts, which is where the conflict and
ordering logic belongs. What only real firmware can prove is that androguard reads the signals
off manifests nobody wrote for us, and that two genuinely different devices sharing 147
packages merge into one row each rather than two — a synthetic pair would share whatever
overlap the fixture author chose.

    UADCLAW_HEAVY_TESTS=1 \\
      UADCLAW_HEAVY_APK_DIR=/path/to/oriole/artifacts \\
      UADCLAW_HEAVY_WORKDIR=/var/tmp/uadclaw-heavy \\
      uv run pytest -n0 -m heavy tests/test_facts_heavy.py

Device 1 is a real Pixel 6 (`oriole-cp2a.260705.006.a1`), an already-extracted
`<partition>/<path-inside-the-image>` tree — the layout `unpack.extract_artifacts` writes, so
`UADCLAW_HEAVY_APK_DIR` points straight at an `artifacts` directory from a previous run and
`unpack.py` regenerates one from the factory zip in about ten minutes. Device 2 is the local
Android 16 emulator system image, unpacked here by the pipeline's own code, whose 228 package
names are recorded in `docs/research/emulator-a16-packages.tsv` from an independent `aapt2`
extraction. That file is the cross-check the design doc asks for: aapt2 and androguard
disagreeing about a package name is a bug signal, and it is asserted rather than assumed.

Every count below was measured on 2026-08-11 and is asserted exactly. A signal that silently
stops being extracted (a namespace typo on `coreApp`, an `<overlay>` lookup that stops
matching) does not raise anything — it returns zero, and only a pinned number notices.
"""

import asyncio
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from androguard.core.apk import APK
from sqlalchemy import func, select
from sqlalchemy import inspect as sa_inspect

from uadclaw.facts import LABEL_UNKNOWN, ApkFacts, parse_apk
from uadclaw.factstore import store_device_facts
from uadclaw.models import PackageFact
from uadclaw.settings import Settings
from uadclaw.stages import apk_paths, artifact_location, parse_device_apks
from uadclaw.unpack import extract_artifacts, unpack_to_partitions

REPO_ROOT = Path(__file__).resolve().parent.parent
EMULATOR_GROUND_TRUTH = REPO_ROOT / "docs" / "research" / "emulator-a16-packages.tsv"
DEFAULT_IMAGE = Path.home() / "Android/Sdk/system-images/android-36.1/google_apis/x86_64/system.img"
EMULATOR_IMAGE = Path(os.environ.get("UADCLAW_HEAVY_IMAGE", str(DEFAULT_IMAGE)))
ORIOLE_ARTIFACTS = os.environ.get("UADCLAW_HEAVY_APK_DIR", "")
REQUIRED_FREE_BYTES = 14 * 1024**3

ORIOLE_DEVICE = "pixel:oriole"
ORIOLE_BUILD = "cp2a.260705.006.a1"
EMULATOR_DEVICE = "google:emulator-a16"
EMULATOR_BUILD = "android-36.1-google_apis-x86_64"

# Measured 2026-08-11 over the whole corpus, per device.
ORIOLE_APKS = 312
EMULATOR_APKS = 228
SHARED_PACKAGES = 147
ALL_PACKAGES = 393
# Of the shared packages, the ones whose signing certificate differs between the two devices.
# Not tampering: v3 key rotation (`com_google_android_gms-rotation-2020`) and the emulator's
# AOSP test key facing the Pixel's Google production key.
CERT_CONFLICTS = 134

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_HEAVY_TESTS") != "1",
        reason="opt-in: set UADCLAW_HEAVY_TESTS=1 (needs both corpora and ~14 GB scratch)",
    ),
]


def signal_counts(facts: list[ApkFacts]) -> dict[str, int]:
    return {
        "apks": len(facts),
        "packages": len({item.package for item in facts}),
        "core_app": sum(item.core_app for item in facts),
        "shared_user_id": sum(bool(item.shared_user_id) for item in facts),
        "persistent": sum(item.persistent for item in facts),
        "overlay": sum(bool(item.overlay_target) for item in facts),
        "static_library": sum(bool(item.static_libraries) for item in facts),
        "auto_generated_rro": sum("auto_generated_rro_" in item.package for item in facts),
        "priv_app": sum(item.priv_app for item in facts),
        "no_code": sum(not item.has_code for item in facts),
        "protected_broadcasts": sum(bool(item.protected_broadcasts) for item in facts),
        "provider_authorities": sum(bool(item.provider_authorities) for item in facts),
        "uses_library_required": sum(bool(item.uses_libraries_required) for item in facts),
        "input_method": sum(item.is_input_method for item in facts),
        "device_admin": sum(item.is_device_admin for item in facts),
        "accessibility": sum(item.is_accessibility_service for item in facts),
        "carrier_service": sum(item.is_carrier_service for item in facts),
    }


@pytest.fixture(scope="module")
def oriole_facts() -> list[ApkFacts]:
    if not ORIOLE_ARTIFACTS:
        pytest.skip("set UADCLAW_HEAVY_APK_DIR to an extracted <partition>/<path> APK tree")
    directory = Path(ORIOLE_ARTIFACTS)
    if not directory.is_dir():
        pytest.skip(f"UADCLAW_HEAVY_APK_DIR is not a directory: {directory}")
    paths = apk_paths(directory)
    parsed, failures = parse_device_apks(directory, paths)
    assert failures == [], "the Pixel corpus parsed 312/312 when this baseline was measured"
    return parsed


@pytest.fixture(scope="module")
def emulator_artifacts(tmp_path_factory) -> Path:
    """Unpacked here by `uadclaw.unpack` rather than by a shell script, so what this test feeds
    androguard is what the pipeline would feed it."""
    if not EMULATOR_IMAGE.is_file():
        pytest.skip(f"emulator system image not on this box: {EMULATOR_IMAGE}")
    work = Path(os.environ.get("UADCLAW_HEAVY_WORKDIR", str(tmp_path_factory.mktemp("facts"))))
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < REQUIRED_FREE_BYTES:
        pytest.skip(f"{work} has {free / 1024**3:.1f} GB free, need ~14 GB")

    settings = Settings(
        postgres_password="test-only-password",
        auth_password="test-only-admin-password",
        session_secret="test-only-session-secret",
    )

    async def run() -> Path:
        partitions = await unpack_to_partitions(EMULATOR_IMAGE, work / "unpack", settings=settings)
        await extract_artifacts(partitions, work / "artifacts")
        # The multi-GB intermediates go the moment the APKs are out, exactly as the unpack
        # stage does it; this test writes ~10 GB and must not leave it on a dev box.
        shutil.rmtree(work / "unpack", ignore_errors=True)
        return work / "artifacts"

    try:
        yield asyncio.run(run())
    finally:
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture(scope="module")
def emulator_facts(emulator_artifacts) -> list[ApkFacts]:
    parsed, failures = parse_device_apks(emulator_artifacts, apk_paths(emulator_artifacts))
    assert failures == [], "the emulator corpus parsed 228/228 when this baseline was measured"
    return parsed


# --- parsing ------------------------------------------------------------------------------


def test_every_pixel_signal_comes_out_at_its_measured_count(oriole_facts):
    """`coreApp` is the one to watch: it is un-namespaced on the `<manifest>` root, so reading
    it through the android namespace returns 0 of 23 and every one of those packages loses the
    strongest boot-critical flag the pipeline has."""
    assert signal_counts(oriole_facts) == {
        "apks": ORIOLE_APKS,
        "packages": 312,
        "core_app": 23,
        "shared_user_id": 44,
        "persistent": 18,
        "overlay": 87,
        "static_library": 1,
        "auto_generated_rro": 27,
        "priv_app": 156,
        "no_code": 95,
        "protected_broadcasts": 17,
        "provider_authorities": 137,
        "uses_library_required": 15,
        "input_method": 4,
        "device_admin": 4,
        "accessibility": 5,
        "carrier_service": 2,
    }


def test_every_emulator_signal_comes_out_at_its_measured_count(emulator_facts):
    assert signal_counts(emulator_facts) == {
        "apks": EMULATOR_APKS,
        "packages": 228,
        "core_app": 20,
        "shared_user_id": 39,
        "persistent": 7,
        "overlay": 102,
        "static_library": 1,
        "auto_generated_rro": 20,
        "priv_app": 75,
        "no_code": 107,
        "protected_broadcasts": 8,
        "provider_authorities": 82,
        "uses_library_required": 7,
        "input_method": 5,
        "device_admin": 3,
        "accessibility": 4,
        "carrier_service": 1,
    }


def test_parse_apk_actually_carries_the_extracted_icon_out(oriole_facts):
    """The `parse_apk` -> `extract_icon` wiring, which only a real APK can reach: nothing in
    the default suite constructs one, so blanking both assignments in `parse_apk` leaves the
    whole suite green.

    Counts measured 2026-08-12 over `oriole-cp2a.260705.006.a1` at `ICON_MAX_DPI = 320`: 52
    rasters, 12 binary-XML drawables this renderer can turn into an SVG, and one 69,038-byte
    PNG refused by the 64 KB cap. The exact numbers rather than "some": a renderer that
    silently stops handling a drawable shape returns fewer icons and raises nothing.
    """
    by_mime: dict[str | None, int] = {}
    for facts in oriole_facts:
        by_mime[facts.icon_mime] = by_mime.get(facts.icon_mime, 0) + 1

    assert by_mime == {None: 248, "image/png": 47, "image/webp": 5, "image/svg+xml": 12}
    assert all((facts.icon_bytes is None) == (facts.icon_mime is None) for facts in oriole_facts), (
        "the two icon columns are written as a pair or not at all"
    )
    assert max(len(facts.icon_bytes) for facts in oriole_facts if facts.icon_bytes) <= 64 * 1024


def test_the_content_uri_scan_yields_its_measured_count(emulator_artifacts, emulator_facts):
    """The dex string-table scan's yield on the emulator corpus, measured 2026-08-14: 48 of
    228 packages reference at least one `content://` authority, 146 distinct authority
    strings, and the top of the list is GMS plumbing (`com.google.android.gsf.gservices` on
    33 packages, `com.google.android.gms.phenotype` on 28).

    The no-code half is the corpus fact rather than a gate test: measured 0 of 107
    resource-only APKs carry any dex member, so their scan result is empty whether or not
    `parse_apk`'s `has_code` gate skips the call — the gate is unobservable on real data
    and the pin says so by checking the members directly."""
    carrying = [facts for facts in emulator_facts if facts.content_uri_authorities]

    assert len(carrying) == 48
    assert len({a for facts in carrying for a in facts.content_uri_authorities}) == 146
    assert (
        sum(
            "com.google.android.gsf.gservices" in facts.content_uri_authorities
            for facts in carrying
        )
        == 33
    )
    assert all(
        facts.content_uri_authorities == tuple(sorted(set(facts.content_uri_authorities)))
        for facts in carrying
    ), "the scan dedupes and sorts"
    no_code = [facts for facts in emulator_facts if not facts.has_code]
    assert len(no_code) == 107
    assert all(not facts.content_uri_authorities for facts in no_code)
    assert all(
        not list(APK(str(path)).get_all_dex())
        for path, facts in zip(apk_paths(emulator_artifacts), emulator_facts, strict=True)
        if not facts.has_code
    ), "no-code packages carry no dex members to scan"


def test_an_undeclared_label_reads_unknown_and_no_resource_id_reaches_a_name(oriole_facts):
    """The misses are undeclared attributes, not resolution failures, so `unknown` is the
    answer and there is no ARSC fallback to build. The `@7f0…` branch is still asserted: it
    fires zero times on this corpus, and a raw id presented as a package's human name is
    exactly what a triage card and a model prompt must never see."""
    overlays = [item for item in oriole_facts if item.overlay_target]
    apps = [item for item in oriole_facts if not item.overlay_target]

    assert sum(item.label != LABEL_UNKNOWN for item in apps) == 177
    assert len(apps) == 225
    assert sum(item.label != LABEL_UNKNOWN for item in overlays) == 18
    assert len(overlays) == 87
    assert [item.package for item in oriole_facts if item.label.startswith("@")] == []
    assert sum(item.label_unresolved for item in oriole_facts) == 0


def test_androguard_agrees_with_the_aapt2_ground_truth_on_every_package(
    emulator_artifacts, emulator_facts
):
    """The design keeps aapt2 as a local cross-check because a disagreement between the two is
    a bug signal. `docs/research/emulator-a16-packages.tsv` is that aapt2 run, preserved."""
    if not EMULATOR_GROUND_TRUTH.is_file():
        pytest.skip(f"ground truth missing (docs/ is gitignored): {EMULATOR_GROUND_TRUTH}")
    expected = {}
    for row in EMULATOR_GROUND_TRUTH.read_text(encoding="utf-8").splitlines():
        if row.strip():
            package, path = row.split("\t")
            expected[path] = package
    measured = {
        str(path.relative_to(emulator_artifacts)): facts.package
        for path, facts in zip(apk_paths(emulator_artifacts), emulator_facts, strict=True)
    }

    assert len(expected) == EMULATOR_APKS
    assert measured == expected


def test_device_paths_and_partitions_come_from_the_image_not_the_manifest(oriole_facts):
    by_package = {item.package: item for item in oriole_facts}
    framework = by_package["android"]

    assert framework.device_path == "/system/framework/framework-res.apk"
    assert framework.partition == "system"
    assert not [item for item in oriole_facts if item.device_path.startswith("/system/system/")]
    assert {item.partition for item in oriole_facts} == {
        "product",
        "system",
        "system_ext",
        "vendor",
    }


def test_androguard_writes_nothing_to_the_worker_log_while_parsing(oriole_facts, capfd):
    """androguard 4.x logs through loguru: eight APKs emitted 123 KB of DEBUG to the console in
    the baseline run, and stdlib `logging.disable()` does not reach it."""
    directory = Path(ORIOLE_ARTIFACTS)
    capfd.readouterr()
    for path in apk_paths(directory)[:25]:
        partition, device_path = artifact_location(directory, path)
        parse_apk(path, partition=partition, device_path=device_path)
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""


# --- the merge, on two real devices --------------------------------------------------------


async def test_two_real_devices_merge_to_one_row_per_package_counting_both(
    db_env, db_session_factory, oriole_facts, emulator_facts
):
    """147 packages are on both a Pixel 6 and an Android 16 emulator. Each must be ONE row
    reading `device_count == 2`; two rows, or one row reading 1, is the failure the whole
    merge exists to prevent, and it is silent — the triage queue would just rank them wrong."""
    for device, build, facts in (
        (ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts),
        (EMULATOR_DEVICE, EMULATOR_BUILD, emulator_facts),
    ):
        async with db_session_factory() as session, session.begin():
            await store_device_facts(
                session,
                device_key=device,
                build=build,
                facts=facts,
                observed_at=datetime.now(UTC),
            )

    shared = {item.package for item in oriole_facts} & {item.package for item in emulator_facts}
    async with db_session_factory() as session:
        total = await session.scalar(select(func.count()).select_from(PackageFact))
        on_two = await session.scalar(
            select(func.count()).select_from(PackageFact).where(PackageFact.device_count == 2)
        )
        conflicted = (
            (await session.execute(select(PackageFact).where(PackageFact.has_conflict)))
            .scalars()
            .all()
        )
        framework = await session.get(PackageFact, "android")

    assert len(shared) == SHARED_PACKAGES
    assert total == ALL_PACKAGES
    assert on_two == SHARED_PACKAGES
    assert framework.device_count == 2
    assert framework.devices == [EMULATOR_DEVICE, ORIOLE_DEVICE]
    assert framework.partitions == ["system"]
    # Pinned because it is the number that decides how a reviewer reads the flag: a cert
    # conflict on 91% of the shared set is signing-key rotation, not a suspicion signal.
    assert len(conflicted) == CERT_CONFLICTS
    assert {item["field"] for row in conflicted for item in row.conflicts} == {
        "cert_issuer",
        "cert_subject",
    }


async def test_the_same_pixel_build_parsed_twice_yields_identical_facts(
    db_env, db_session_factory, oriole_facts
):
    """The reproducibility anchor, on the real corpus: same bytes in, same 312 rows out, and no
    device counted twice."""
    snapshots = []
    for _ in range(2):
        async with db_session_factory() as session, session.begin():
            await store_device_facts(
                session,
                device_key=ORIOLE_DEVICE,
                build=ORIOLE_BUILD,
                facts=oriole_facts,
                observed_at=datetime.now(UTC),
            )
        async with db_session_factory() as session:
            rows = (
                await session.execute(select(PackageFact).order_by(PackageFact.package))
            ).scalars()
            snapshots.append(
                [
                    {
                        column.key: getattr(row, column.key)
                        for column in sa_inspect(row).mapper.column_attrs
                    }
                    for row in rows
                ]
            )

    assert len(snapshots[0]) == 312
    assert snapshots[0] == snapshots[1]
    assert {row["device_count"] for row in snapshots[0]} == {1}
