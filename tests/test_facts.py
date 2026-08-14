"""Fact merging, conflict flagging and the `extract_facts` stage's guards.

Parsing itself is proven against real firmware in `test_facts_heavy.py`: a synthetic APK would
have to carry a hand-built binary AXML, which would test the fixture rather than androguard.
What is proven here is everything a wrong merge would hide — a shared package splitting into
two rows, a re-scan drifting, a second signing certificate quietly winning — plus the stage's
own input handling, all of which run in milliseconds and none of which need an APK to exist.
"""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from lxml import etree
from sqlalchemy import func, select
from sqlalchemy import inspect as sa_inspect

from uadclaw import icons as icons_module
from uadclaw import jobs as jobs_module
from uadclaw import stages as stages_module
from uadclaw.facts import (
    LABEL_UNKNOWN,
    ApkFacts,
    ApkParseError,
    IntentFilterFact,
    _uses_libraries,
    parse_apk,
)
from uadclaw.factstore import (
    _SCALAR_OBSERVATION_FIELDS,
    UNION_FIELDS,
    FactMergeError,
    merge_observations,
    observation_row,
    store_device_facts,
)
from uadclaw.firmware import FirmwareRef
from uadclaw.models import DeviceScan, JobKind, PackageFact, PackageObservation
from uadclaw.settings import Settings
from uadclaw.stages import (
    PipelineState,
    StageInputError,
    TooManyApkParseFailuresError,
    _check_parse_budget,
    apk_paths,
    artifact_location,
    extract_facts_stage,
    read_state,
    write_state,
)
from uadclaw.worker import StageContext

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def make_facts(package: str, **overrides) -> ApkFacts:
    values: dict[str, object] = {
        "package": package,
        "label": "Example",
        "label_unresolved": False,
        "version_code": 1,
        "partition": "system",
        "device_path": f"/system/priv-app/{package}/{package}.apk",
        "priv_app": True,
        "sha256": "0" * 64,
        "cert_issuer": "Organization: Google Inc.",
        "cert_subject": "Organization: Google Inc.",
        "core_app": False,
        "shared_user_id": None,
        "persistent": False,
        "has_code": True,
        "overlay_target": None,
        "overlay_static": False,
        "overlay_priority": None,
    }
    values.update(overrides)
    return ApkFacts(**values)  # type: ignore[arg-type]


def rows_for(*observations: tuple[ApkFacts, str, str]) -> list[dict]:
    return [
        observation_row(facts, device_key=device, build=build, observed_at=NOW)
        for facts, device, build in observations
    ]


def as_dict(instance) -> dict:
    columns = sa_inspect(instance).mapper.column_attrs
    return {column.key: getattr(instance, column.key) for column in columns}


async def store(session_factory, device_key: str, build: str, *facts: ApkFacts) -> None:
    async with session_factory() as session, session.begin():
        await store_device_facts(
            session, device_key=device_key, build=build, facts=facts, observed_at=NOW
        )


async def merged_row(session_factory, package: str) -> PackageFact:
    async with session_factory() as session:
        fact = await session.get(PackageFact, package)
    assert fact is not None, f"no merged row for {package}"
    return fact


# --- the merge, as a pure function --------------------------------------------------------


def test_merge_refuses_to_fold_two_package_names_into_one_row():
    rows = rows_for(
        (make_facts("com.a"), "pixel:oriole", "A.1"),
        (make_facts("com.b"), "pixel:oriole", "A.1"),
    )

    with pytest.raises(FactMergeError):
        merge_observations(rows)


def test_the_merge_does_not_depend_on_scan_order():
    """The whole point of recomputing from every observation rather than folding into
    whatever was already there: `reversed` must not change one field."""
    rows = rows_for(
        (make_facts("com.a", core_app=True, protected_broadcasts=("X",)), "pixel:oriole", "A.1"),
        (make_facts("com.a", persistent=True, protected_broadcasts=("Y",)), "pixel:raven", "B.2"),
        (make_facts("com.a", version_code=9), "xiaomi:lisa", "C.3"),
    )

    assert merge_observations(rows) == merge_observations(list(reversed(rows)))


def test_one_device_saying_nothing_is_not_a_conflict():
    """Absence of evidence is not contradiction: a build that never declared a sharedUserId
    must not flag every device that did."""
    rows = rows_for(
        (make_facts("com.a", shared_user_id="android.uid.system"), "pixel:oriole", "A.1"),
        (make_facts("com.a", shared_user_id=None), "pixel:raven", "B.2"),
    )

    merged = merge_observations(rows)

    assert merged["has_conflict"] is False
    assert merged["shared_user_id"] == "android.uid.system"


def test_a_declared_label_beats_an_undeclared_one_whichever_device_carries_it():
    rows = rows_for(
        (make_facts("com.a", label=LABEL_UNKNOWN), "pixel:aaa", "A.1"),
        (make_facts("com.a", label="Calculator"), "pixel:zzz", "B.2"),
    )

    assert merge_observations(rows)["label"] == "Calculator"


def test_the_icon_of_the_lowest_ordered_observation_wins():
    """Two devices shipping different artwork for one package is a tie the merge has to break
    the same way every time, or a re-scan changes the dashboard for no reason."""
    rows = rows_for(
        (make_facts("com.a", icon_bytes=b"zzz-icon", icon_mime="image/png"), "pixel:zzz", "B.2"),
        (make_facts("com.a", icon_bytes=b"aaa-icon", icon_mime="image/webp"), "pixel:aaa", "A.1"),
    )

    merged = merge_observations(rows)

    assert (merged["icon_bytes"], merged["icon_mime"]) == (b"aaa-icon", "image/webp")
    assert merge_observations(list(reversed(rows))) == merged


def test_a_device_that_shipped_no_icon_does_not_blank_one_that_did():
    """Strict first-wins would let the alphabetically-first device decide there is no icon —
    the same trap `label` names, and the reason both rules skip the silent observations."""
    rows = rows_for(
        (make_facts("com.a"), "pixel:aaa", "A.1"),
        (make_facts("com.a", icon_bytes=b"icon", icon_mime="image/png"), "pixel:zzz", "B.2"),
    )

    merged = merge_observations(rows)

    assert (merged["icon_bytes"], merged["icon_mime"]) == (b"icon", "image/png")


def test_a_package_no_device_shipped_an_icon_for_merges_to_nothing():
    """The majority case: 223 of the 312 APKs on the Pixel corpus declare no resolvable icon,
    so the dashboard's monogram fallback is the designed path and not an error state."""
    merged = merge_observations(rows_for((make_facts("com.a"), "pixel:aaa", "A.1")))

    assert merged["icon_bytes"] is None
    assert merged["icon_mime"] is None


def test_every_observation_column_is_either_upserted_or_deliberately_not():
    """The re-scan upsert's `set_` list, pinned as a CONTRACT rather than per column.

    `test_the_same_build_parsed_twice_yields_identical_facts` writes the same values twice, so
    it cannot tell that list from an empty one — which is how the icon columns were omitted
    from it and nothing went red. Fixing that for two columns leaves the next column added
    here with the identical silent gap, so the set difference is asserted instead: a new
    column lands on the left of this subtraction and reds until somebody either upserts it or
    writes down why not.

    The four exclusions each have a reason. `id` is the surrogate key; `device_key` and
    `build` are two thirds of the conflict key, so re-asserting them says nothing; and
    `observed_at` is excluded on purpose — it is when the APK was FIRST seen, per the comment
    on the `set_` itself, so a re-scan must not move it.
    """
    columns = {column.key for column in sa_inspect(PackageObservation).mapper.column_attrs}
    upserted = {*_SCALAR_OBSERVATION_FIELDS, *UNION_FIELDS, "package"}

    assert columns - upserted == {"id", "device_key", "build", "observed_at"}
    assert upserted - columns == set(), "the upsert names a column that does not exist"


def test_first_seen_and_last_seen_span_the_observations():
    later = NOW + timedelta(days=30)
    rows = [
        observation_row(make_facts("com.a"), device_key="pixel:a", build="A.1", observed_at=NOW),
        observation_row(
            make_facts("com.a", device_path="/product/app/A/A.apk"),
            device_key="pixel:b",
            build="B.2",
            observed_at=later,
        ),
    ]

    merged = merge_observations(rows)

    assert (merged["first_seen_at"], merged["last_seen_at"]) == (NOW, later)


def test_the_highest_version_code_survives_the_merge():
    rows = rows_for(
        (make_facts("com.a", version_code=11), "pixel:oriole", "A.1"),
        (make_facts("com.a", version_code=None), "pixel:raven", "B.2"),
        (make_facts("com.a", version_code=7), "xiaomi:lisa", "C.3"),
    )

    assert merge_observations(rows)["version_code"] == 11


# --- the merge, through Postgres ----------------------------------------------------------


async def test_two_devices_sharing_a_package_produce_one_row_counting_two_devices(
    db_env, db_session_factory
):
    """Task 4's verify line, in miniature; `test_facts_heavy.py` proves the same thing on the
    two real corpora."""
    await store(
        db_session_factory,
        "pixel:oriole",
        "cp2a.260705.006.a1",
        make_facts("com.android.shared"),
        make_facts("com.oriole.only"),
    )
    await store(
        db_session_factory,
        "pixel:raven",
        "bp1a.250505.005",
        make_facts("com.android.shared"),
        make_facts("com.raven.only"),
    )

    async with db_session_factory() as session:
        total = await session.scalar(select(func.count()).select_from(PackageFact))
    shared = await merged_row(db_session_factory, "com.android.shared")

    assert total == 3, "one row per package name, not one per (device, package)"
    assert shared.device_count == 2
    assert shared.devices == ["pixel:oriole", "pixel:raven"]
    assert (await merged_row(db_session_factory, "com.oriole.only")).device_count == 1


async def test_the_same_build_parsed_twice_yields_identical_facts(db_env, db_session_factory):
    """The reproducibility claim. Re-scanning must upsert onto the same observation rows and
    recompute to the same merged values — not append a second device, not move a timestamp."""
    facts = (make_facts("com.android.shared"), make_facts("com.example.two"))
    snapshots = []
    for _ in range(2):
        await store(db_session_factory, "pixel:oriole", "cp2a.260705.006.a1", *facts)
        async with db_session_factory() as session:
            rows = (
                await session.execute(select(PackageFact).order_by(PackageFact.package))
            ).scalars()
            observations = await session.scalar(
                select(func.count()).select_from(PackageObservation)
            )
        snapshots.append(([as_dict(row) for row in rows], observations))

    assert snapshots[0] == snapshots[1]
    assert snapshots[0][1] == 2, "a re-scan upserts its observations rather than adding more"
    assert [row["device_count"] for row in snapshots[0][0]] == [1, 1]


async def test_an_icon_survives_the_round_trip_through_postgres(db_env, db_session_factory):
    """The bytes cross a BYTEA column and a recompute, and the recompute is what would drop
    them: it rebuilds the merged row from the observations rather than leaving the old one."""
    await store(
        db_session_factory,
        "pixel:oriole",
        "cp2a.260705.006.a1",
        make_facts("com.example.app", icon_bytes=b"\x89PNG\r\n\x1a\nbody", icon_mime="image/png"),
    )

    fact = await merged_row(db_session_factory, "com.example.app")

    assert fact.icon_bytes == b"\x89PNG\r\n\x1a\nbody"
    assert fact.icon_mime == "image/png"


async def test_a_rescan_replaces_the_icon_the_first_scan_stored(db_env, db_session_factory):
    """A re-scan exists so a parser fix can land, and the icon renderer is the newest thing
    that can learn to read a drawable it used to refuse. Same device, same build, same path:
    the upsert's update set is the only thing deciding whether the new artwork arrives, and a
    re-scan carrying identical values cannot tell that set from an empty one."""
    await store(db_session_factory, "pixel:oriole", "A.1", make_facts("com.a"))
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts("com.a", icon_bytes=b"<svg/>", icon_mime="image/svg+xml"),
    )

    fact = await merged_row(db_session_factory, "com.a")

    assert (fact.icon_bytes, fact.icon_mime) == (b"<svg/>", "image/svg+xml")


async def test_a_second_signing_certificate_is_flagged_and_both_values_are_kept(
    db_env, db_session_factory
):
    """The named conflict case. The row survives — flagged, with every value and the device
    that carried it — because dropping it would hide the disagreement instead of surfacing it,
    and silently taking the newer certificate would let a repackaged app inherit the real
    one's rating."""
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts("com.contested", cert_issuer="Organization: Google Inc."),
    )
    await store(
        db_session_factory,
        "xiaomi:lisa",
        "B.2",
        make_facts("com.contested", cert_issuer="Organization: Definitely Not Google"),
    )

    fact = await merged_row(db_session_factory, "com.contested")

    assert fact.has_conflict is True
    assert fact.device_count == 2
    issuer_conflict = next(item for item in fact.conflicts if item["field"] == "cert_issuer")
    assert [entry["value"] for entry in issuer_conflict["values"]] == [
        "Organization: Definitely Not Google",
        "Organization: Google Inc.",
    ]
    assert [entry["devices"] for entry in issuer_conflict["values"]] == [
        ["xiaomi:lisa"],
        ["pixel:oriole"],
    ]
    # Deterministic by (device_key, build, device_path), never "whichever scan ran last".
    assert fact.cert_issuer == "Organization: Google Inc."


async def test_a_danger_flag_on_one_device_survives_the_merge(db_env, db_session_factory):
    """`coreApp` disagreeing across devices is both a conflict and a floor input. The merge
    must take the restrictive side: a rule ladder reading `core_app=False` here would compute
    `Recommended` for a package one of these devices will not boot without."""
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts("com.core", core_app=True, persistent=True),
    )
    await store(
        db_session_factory,
        "pixel:raven",
        "B.2",
        make_facts("com.core", core_app=False, persistent=False),
    )

    fact = await merged_row(db_session_factory, "com.core")

    assert fact.core_app is True
    assert fact.persistent is True
    assert {item["field"] for item in fact.conflicts} == {"core_app", "persistent"}


async def test_list_signals_are_unioned_across_devices(db_env, db_session_factory):
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts(
            "com.lib",
            protected_broadcasts=("com.x.BOOT",),
            uses_libraries_required=("android.ext.shared",),
            intent_filters=(
                IntentFilterFact(
                    component="activity",
                    component_class="com.lib.Home",
                    actions=("android.intent.action.MAIN",),
                    categories=("android.intent.category.HOME",),
                    priority=100,
                ),
            ),
        ),
    )
    await store(
        db_session_factory,
        "xiaomi:lisa",
        "B.2",
        make_facts(
            "com.lib",
            partition="product",
            device_path="/product/app/Lib/Lib.apk",
            protected_broadcasts=("com.y.SYNC",),
            uses_libraries_required=("android.ext.shared",),
        ),
    )

    fact = await merged_row(db_session_factory, "com.lib")

    assert fact.protected_broadcasts == ["com.x.BOOT", "com.y.SYNC"]
    assert fact.uses_libraries_required == ["android.ext.shared"], "the union deduplicates"
    assert fact.partitions == ["product", "system"]
    assert [item["priority"] for item in fact.intent_filters] == [100]


async def test_one_device_shipping_a_package_twice_stays_two_observations(
    db_env, db_session_factory
):
    """Two APKs on one device declaring one package name is rare but real, and it is the same
    class of problem as two devices disagreeing. One row per APK means it flows through the
    ordinary conflict check instead of being collapsed before anything compared them."""
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts("com.twice", device_path="/system/priv-app/A/A.apk"),
        make_facts(
            "com.twice",
            partition="product",
            device_path="/product/app/B/B.apk",
            cert_issuer="Organization: Somebody Else",
        ),
    )

    fact = await merged_row(db_session_factory, "com.twice")

    assert fact.device_count == 1, "two APKs on one phone are still one device"
    assert fact.has_conflict is True
    async with db_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(PackageObservation)) == 2


# --- the manifest parse -------------------------------------------------------------------


def test_a_uses_static_library_consumer_is_a_hard_dependency():
    """The consumer half of the library edge class. `<uses-static-library>` carries no
    `required` attribute, so a static consumer is a hard dependency by construction: its name
    lands in the bucket a `required="true"` uses-library uses, and it leaves the optional
    bucket, whose entry would claim the app works without a library the same manifest makes
    unwaivable. Both local corpora declare zero of these elements, so no real manifest can pin
    this branch — a synthetic element can, because what is under test is this module's
    element-walking rather than androguard's AXML decode (which stays the heavy suite's job)."""
    application = etree.fromstring(
        '<application xmlns:android="http://schemas.android.com/apk/res/android">'
        '<uses-library android:name="android.ext.shared" android:required="false"/>'
        '<uses-library android:name="com.framework.required"/>'
        '<uses-library android:name="com.conflicted" android:required="false"/>'
        '<uses-static-library android:name="com.google.android.trichromelibrary" '
        'android:version="435302154"/>'
        '<uses-static-library android:name="com.framework.required" android:version="1"/>'
        '<uses-static-library android:name="com.conflicted" android:version="1"/>'
        "</application>"
    )

    required, optional = _uses_libraries(application)

    assert required == (
        "com.framework.required",
        "com.google.android.trichromelibrary",
        "com.conflicted",
    )
    assert optional == ("android.ext.shared",)


# --- the stage's own inputs ---------------------------------------------------------------


def test_artifact_location_collapses_the_nested_system_root(tmp_path):
    """`system.img` entries carry a nested `system/` root while `product` is rooted directly at
    `app/`. Both must land on one spelling of the device path, or the same file on two
    partitions reads as two different files."""
    assert artifact_location(tmp_path, tmp_path / "system/system/priv-app/A/A.apk") == (
        "system",
        "/system/priv-app/A/A.apk",
    )
    assert artifact_location(tmp_path, tmp_path / "product/app/B/B.apk") == (
        "product",
        "/product/app/B/B.apk",
    )


def test_artifact_location_refuses_a_file_with_no_partition_directory(tmp_path):
    with pytest.raises(StageInputError, match="partition directory"):
        artifact_location(tmp_path, tmp_path / "loose.apk")


def test_apk_paths_finds_every_apk_and_refuses_to_walk_a_symlink(tmp_path):
    """A firmware image's own symlink must not walk the host filesystem into the corpus."""
    artifacts = tmp_path / "artifacts"
    (artifacts / "system/priv-app/A").mkdir(parents=True)
    (artifacts / "system/priv-app/A/A.apk").write_bytes(b"a")
    (artifacts / "system/etc/permissions").mkdir(parents=True)
    (artifacts / "system/etc/permissions/p.xml").write_bytes(b"<x/>")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "elsewhere.apk").write_bytes(b"nope")
    (artifacts / "system/link").symlink_to(outside, target_is_directory=True)

    assert apk_paths(artifacts) == [artifacts / "system/priv-app/A/A.apk"]


def test_one_unreadable_apk_out_of_many_is_recorded_rather_than_fatal():
    _check_parse_budget(312, [{"device_path": "/system/app/X/X.apk", "error": "bad"}], 0.05)


def test_a_mostly_unparseable_device_fails_naming_the_first_bad_apk():
    failures = [{"device_path": f"/system/app/X{i}/X.apk", "error": "bad"} for i in range(40)]

    with pytest.raises(TooManyApkParseFailuresError) as excinfo:
        _check_parse_budget(100, failures, 0.05)

    assert "/system/app/X0/X.apk" in str(excinfo.value)
    assert "40 of 100" in str(excinfo.value)


def test_a_parse_failure_budget_of_one_is_refused_at_settings_load():
    with pytest.raises(ValueError, match="max_apk_parse_failure_ratio"):
        Settings(
            postgres_password="p",
            auth_password="a",
            session_secret="s",
            max_apk_parse_failure_ratio=1.0,
        )


def test_a_file_that_is_not_an_apk_fails_as_bad_input_naming_the_device_path(tmp_path):
    """Bad input gets its own type so the stage can record it and carry on; a genuine defect
    here would raise something else and take the job down, which is the intent."""
    path = tmp_path / "Broken.apk"
    path.write_bytes(b"PK\x03\x04not really a zip")

    with pytest.raises(ApkParseError) as excinfo:
        parse_apk(path, partition="system", device_path="/system/app/Broken/Broken.apk")

    assert "/system/app/Broken/Broken.apk" in str(excinfo.value)


# --- the stage end to end -----------------------------------------------------------------


async def queued_job(session_factory) -> uuid.UUID:
    """A real job row: `device_scans.job_id` is a foreign key, so which job produced a scan
    survives as a constraint rather than as a convention."""
    async with session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session,
            kind=JobKind.FIRMWARE_ANALYSIS.value,
            params={"driver": "pixel", "device": "oriole"},
        )
        return job.id


def build_scratch(tmp_path: Path, apk_names: list[str]) -> Path:
    """A scratch directory in the shape `unpack` leaves behind: a state file plus
    `artifacts/<partition>/<path-inside-the-image>`."""
    scratch = tmp_path / "scratch"
    write_state(
        scratch,
        PipelineState(
            ref=FirmwareRef(
                driver="pixel", device="oriole", build="A.1", url="https://example.invalid/a.zip"
            ),
            partitions=["system"],
            artifact_count=len(apk_names) + 1,
            apk_count=len(apk_names),
        ),
    )
    for name in apk_names:
        apk = scratch / f"artifacts/system/system/priv-app/{name}/{name}.apk"
        apk.parent.mkdir(parents=True, exist_ok=True)
        apk.write_bytes(b"pretend APK")
    config = scratch / "artifacts/system/system/etc/permissions/privapp-permissions.xml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_bytes(b"<permissions/>")
    return scratch


async def test_the_stage_stores_facts_then_deletes_the_apks_and_keeps_the_config_inputs(
    db_env, db_session_factory, monkeypatch, tmp_path
):
    """Retention is facts-only (decision 5) and it is enforced by this stage, not at the end
    of the job: the APKs go the moment Postgres has their facts. The config XMLs stay, because
    `corpus_graph` and `rule_ladder` read them and they are kilobytes.

    `parse_apk` is stubbed because a synthetic APK would need a hand-built binary AXML; real
    parsing is pinned in `test_facts_heavy.py`. What is under test here is the wiring.
    """
    scratch = build_scratch(tmp_path, ["Alpha", "Beta"])
    monkeypatch.setattr(
        stages_module,
        "parse_apk",
        lambda path, *, partition, device_path: make_facts(
            f"com.example.{path.stem.lower()}", partition=partition, device_path=device_path
        ),
    )
    ctx = StageContext(
        job_id=await queued_job(db_session_factory),
        attempt=1,
        scratch_dir=scratch,
        session_factory=db_session_factory,
    )

    await extract_facts_stage(ctx)

    async with db_session_factory() as session:
        packages = (
            await session.execute(select(PackageFact.package).order_by(PackageFact.package))
        ).scalars()
        scan = (await session.execute(select(DeviceScan))).scalar_one()
    assert list(packages) == ["com.example.alpha", "com.example.beta"]
    assert (scan.device_key, scan.apk_total, scan.parsed_ok, scan.parse_failed) == (
        "pixel:oriole",
        2,
        2,
        0,
    )
    assert apk_paths(scratch / "artifacts") == []
    assert (scratch / "artifacts/system/system/etc/permissions/privapp-permissions.xml").is_file()
    assert read_state(scratch).package_count == 2


async def test_one_unparseable_apk_lands_in_the_scan_row_and_the_rest_still_store(
    db_env, db_session_factory, monkeypatch, tmp_path
):
    """The observable half of the failure policy: retention deletes the APK, so a failure that
    was only ever logged would be unreconstructable afterwards."""
    monkeypatch.setenv("MAX_APK_PARSE_FAILURE_RATIO", "0.6")
    scratch = build_scratch(tmp_path, ["Alpha", "Broken"])

    def fake_parse(path: Path, *, partition: str, device_path: str) -> ApkFacts:
        if "Broken" in path.name:
            raise ApkParseError(f"androguard could not open {device_path}")
        return make_facts("com.example.alpha", partition=partition, device_path=device_path)

    monkeypatch.setattr(stages_module, "parse_apk", fake_parse)
    ctx = StageContext(
        job_id=await queued_job(db_session_factory),
        attempt=1,
        scratch_dir=scratch,
        session_factory=db_session_factory,
    )

    await extract_facts_stage(ctx)

    async with db_session_factory() as session:
        scan = (await session.execute(select(DeviceScan))).scalar_one()
        stored = (await session.execute(select(PackageFact.package))).scalars().all()
    assert (scan.apk_total, scan.parsed_ok, scan.parse_failed) == (2, 1, 1)
    assert scan.failures == [
        {
            "device_path": "/system/priv-app/Broken/Broken.apk",
            "error": "androguard could not open /system/priv-app/Broken/Broken.apk",
        }
    ]
    assert stored == ["com.example.alpha"]
    assert read_state(scratch).parse_failed_count == 1


async def test_an_icon_that_blows_up_costs_its_own_apk_nothing_and_the_device_nothing(
    db_env, db_session_factory, monkeypatch, tmp_path
):
    """An icon is decoration, and `parse_device_apks` catches `ApkParseError` alone — so
    anything else escaping the icon path does not fail one APK, it fails the whole stage and
    discards every other package on the device.

    A plain `ValueError` on purpose. If this pinned `RecursionError`, a guard spelled
    `except RecursionError` would pass it while `MemoryError`, a future androguard's
    `AttributeError` and every other escape still killed the stage.

    The observable is stronger than "the other APKs survived": the hostile APK keeps its own
    facts too, and the failure budget sees nothing, because there was no fact-parse failure
    to see."""
    scratch = build_scratch(tmp_path, ["Alpha", "Hostile", "Beta"])

    def parse_with_icon(path: Path, *, partition: str, device_path: str) -> ApkFacts:
        """`parse_apk`'s own shape: build the facts, then read the icon off the open APK.
        The real `extract_icon` runs — only the APK it is handed is a stand-in."""
        apk = _ApkWhoseIconExplodes() if "Hostile" in path.name else _ApkWithNoIcon()
        icon = icons_module.extract_icon(apk, origin=device_path)
        return make_facts(
            f"com.example.{path.stem.lower()}",
            partition=partition,
            device_path=device_path,
            icon_bytes=icon[0] if icon else None,
            icon_mime=icon[1] if icon else None,
        )

    monkeypatch.setattr(stages_module, "parse_apk", parse_with_icon)
    ctx = StageContext(
        job_id=await queued_job(db_session_factory),
        attempt=1,
        scratch_dir=scratch,
        session_factory=db_session_factory,
    )

    await extract_facts_stage(ctx)

    async with db_session_factory() as session:
        rows = (
            (await session.execute(select(PackageFact).order_by(PackageFact.package)))
            .scalars()
            .all()
        )
        scan = (await session.execute(select(DeviceScan))).scalar_one()
    assert [row.package for row in rows] == [
        "com.example.alpha",
        "com.example.beta",
        "com.example.hostile",
    ]
    assert {row.package: row.icon_bytes for row in rows}["com.example.hostile"] is None
    assert (scan.apk_total, scan.parsed_ok, scan.parse_failed) == (3, 3, 0)
    assert scan.failures == []
    assert read_state(scratch).parse_failed_count == 0


class _ApkWhoseIconExplodes:
    def get_app_icon(self, max_dpi: int = 65536):
        raise ValueError("androguard fell over rendering a drawable")


class _ApkWithNoIcon:
    def get_app_icon(self, max_dpi: int = 65536):
        return None


async def test_the_stage_refuses_a_scratch_directory_with_no_artifacts(
    db_env, db_session_factory, tmp_path
):
    scratch = build_scratch(tmp_path, [])
    ctx = StageContext(
        job_id=await queued_job(db_session_factory),
        attempt=1,
        scratch_dir=scratch,
        session_factory=db_session_factory,
    )

    with pytest.raises(StageInputError, match="holds no APK"):
        await extract_facts_stage(ctx)


def test_androguard_loguru_records_are_dropped_by_the_library_not_by_a_fixture():
    """androguard 4.x logs through loguru, which stdlib `logging.disable()` does not reach;
    eight APKs emitted 123 KB of DEBUG in the baseline run. Silencing it lives in
    `uadclaw.facts`, so the worker does not flood its own logs in production only — and it is
    scoped to androguard rather than removing every sink the process owns.

    loguru picks the record's name off the calling frame's `__name__`, so exec'ing under a
    borrowed module name is what actually exercises the filter.
    """
    from loguru import logger

    import uadclaw.facts  # noqa: F401  (importing it is what installs the filter)

    captured: list[str] = []
    sink = logger.add(captured.append, level="DEBUG", format="{message}")
    try:
        exec(
            "logger.debug('123 KB of DEBUG')", {"__name__": "androguard.core.apk", "logger": logger}
        )
        exec("logger.debug('from anywhere else')", {"__name__": "uadclaw.probe", "logger": logger})
    finally:
        logger.remove(sink)

    assert any("anywhere else" in line for line in captured)
    assert not [line for line in captured if "123 KB" in line]
