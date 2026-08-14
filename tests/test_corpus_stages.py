"""The deterministic core against a real database: the `/etc` parsing and the three stages.

The graph, the ladder and the filter are pinned as pure functions elsewhere. What only a
database proves is the wiring, and the wiring has one shape that is easy to get wrong and
silent when it is: three stages write disjoint columns of one row, so a stage that names every
column blanks the other two stages' output on every run, and the next reader sees a package
with no floor rather than an error.

Everything here runs on synthetic `package_facts` rows rather than APKs, deliberately: task 4
ended up with its signal extraction pinned only by heavy tests, so a parsing regression reads
green in CI. The inputs of this milestone are database rows, so there is no excuse for that
here — the real corpora add measured counts in `test_corpus_heavy.py`, they do not carry the
logic.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from uadclaw import jobs as jobs_module
from uadclaw.corpusstore import CorpusStoreError, load_config_inputs, require_corpus
from uadclaw.etcconfig import (
    ConfigInputs,
    merge_config_inputs,
    parse_config_inputs,
)
from uadclaw.facts import ApkFacts, IntentFilterFact, LibraryFact
from uadclaw.factstore import record_device_scan, store_device_facts
from uadclaw.ladder import FloorRule, Removal
from uadclaw.models import DeviceScan, JobKind, PackageAnalysis
from uadclaw.stages import (
    StageInputError,
    corpus_graph_stage,
    filter_stage,
    rule_ladder_stage,
)
from uadclaw.upstream import UpstreamListError
from uadclaw.worker import StageContext

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
PIXEL = "pixel:oriole"
EMULATOR = "google:emulator-a16"

UPSTREAM_JSON = """
{"com.listed.one": {"list": "Google", "removal": "Recommended"},
 "com.android.settings": {"list": "Google", "removal": "Unsafe"}}
"""

PRIVAPP_XML = """<?xml version="1.0" encoding="utf-8"?>
<permissions>
    <privapp-permissions package="com.example.allowlisted">
        <permission name="android.permission.REBOOT"/>
        <permission name="android.permission.MANAGE_USERS"/>
        <deny-permission name="android.permission.CLEAR_APP_CACHE"/>
    </privapp-permissions>
    <library name="android.test.base" file="/system/framework/android.test.base.jar"/>
</permissions>
"""

ROLES_XML = """<?xml version="1.0" encoding="utf-8"?>
<roles>
    <role name="android.app.role.SMS" static="true" defaultHolders="com.example.messages"/>
    <role name="android.app.role.BROWSER" defaultHolders="com.example.browser"/>
</roles>
"""


def make_facts(package: str, **overrides) -> ApkFacts:
    values: dict[str, object] = {
        "package": package,
        "label": "Example",
        "label_unresolved": False,
        "version_code": 1,
        "partition": "system",
        "device_path": f"/system/app/{package}/{package}.apk",
        "priv_app": False,
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


CORPUS = [
    make_facts("com.listed.one"),
    make_facts("com.android.settings", core_app=True),
    make_facts("com.android.settings.overlay.oriole", overlay_target="com.android.settings"),
    make_facts("com.example.allowlisted"),
    make_facts("com.vendor.lib", libraries=(LibraryFact(name="com.vendor.support"),)),
    make_facts("com.vendor.app", uses_libraries_required=("com.vendor.support",)),
    make_facts(
        "com.example.messages",
        intent_filters=(
            IntentFilterFact(
                component="receiver",
                component_class="Sms",
                actions=("android.provider.Telephony.SMS_DELIVER",),
                categories=(),
            ),
        ),
    ),
    make_facts("com.example.contentprovider", provider_authorities=("com.example.content",)),
    make_facts(
        "com.example.contentclient",
        content_uri_authorities=("com.example.content", "com.example.gone"),
    ),
]


async def seed_job(session_factory) -> uuid.UUID:
    async with session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session,
            kind=JobKind.FIRMWARE_ANALYSIS.value,
            params={"driver": "pixel", "device": "oriole"},
        )
        return job.id


async def seed_corpus(session_factory, job_id: uuid.UUID, *, device_key: str = PIXEL) -> None:
    async with session_factory() as session, session.begin():
        await record_device_scan(
            session,
            job_id=job_id,
            device_key=device_key,
            build="cp2a.260705.006.a1",
            scanned_at=NOW,
            apk_total=len(CORPUS),
            parsed_ok=len(CORPUS),
            failures=[],
        )
        await store_device_facts(
            session,
            device_key=device_key,
            build="cp2a.260705.006.a1",
            facts=CORPUS,
            observed_at=NOW,
        )


def build_scratch(tmp_path: Path) -> Path:
    """Scratch in the shape `unpack` leaves it: the config XMLs the ladder reads survive
    retention beside the (already deleted) APKs."""
    scratch = tmp_path / "scratch"
    permissions = scratch / "artifacts/system/system/etc/permissions"
    permissions.mkdir(parents=True, exist_ok=True)
    (permissions / "privapp-permissions-platform.xml").write_text(PRIVAPP_XML, encoding="utf-8")
    (permissions / "roles.xml").write_text(ROLES_XML, encoding="utf-8")
    return scratch


@pytest.fixture
def upstream_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "uad_lists.json"
    path.write_text(UPSTREAM_JSON, encoding="utf-8")
    monkeypatch.setenv("UPSTREAM_LIST_PATH", str(path))
    return path


async def context(session_factory, tmp_path) -> StageContext:
    return StageContext(
        job_id=await seed_job(session_factory),
        attempt=1,
        scratch_dir=build_scratch(tmp_path),
        session_factory=session_factory,
    )


async def analysis(session_factory) -> dict[str, PackageAnalysis]:
    async with session_factory() as session:
        rows = (await session.execute(select(PackageAnalysis))).scalars().all()
    return {row.package: row for row in rows}


# --- the /etc rule inputs ---------------------------------------------------------------------


def test_the_config_parser_reads_allowlists_roles_and_platform_libraries(tmp_path):
    artifacts = build_scratch(tmp_path) / "artifacts"

    config = parse_config_inputs(artifacts)

    assert config.privapp_permissions == {
        "com.example.allowlisted": (
            "android.permission.MANAGE_USERS",
            "android.permission.REBOOT",
        )
    }
    assert config.static_role_holders == {"android.app.role.SMS": ("com.example.messages",)}
    assert config.platform_libraries == frozenset({"android.test.base"})
    assert config.files_read == 2
    assert config.files_failed == ()


def test_a_denied_permission_is_not_counted_as_an_integration_signal(tmp_path):
    """The score is "how deeply is this wired into the platform"; an explicitly denied
    permission is the opposite of that."""
    config = parse_config_inputs(build_scratch(tmp_path) / "artifacts")

    assert config.permission_count("com.example.allowlisted") == 2
    assert (
        "android.permission.CLEAR_APP_CACHE"
        not in config.privapp_permissions["com.example.allowlisted"]
    )


def test_a_non_static_role_is_not_a_ladder_input(tmp_path):
    """A reassignable role has a user-selectable alternative by definition, so its holder is
    not pinned by it."""
    config = parse_config_inputs(build_scratch(tmp_path) / "artifacts")

    assert "android.app.role.BROWSER" not in config.static_role_holders


def test_a_config_xml_declaring_a_dtd_is_refused_unparsed(tmp_path):
    """Entity expansion is the whole billion-laughs class and no AOSP config XML carries a
    DTD, so the document is refused rather than handed to the parser."""
    artifacts = build_scratch(tmp_path) / "artifacts"
    bomb = artifacts / "system/system/etc/permissions/bomb.xml"
    bomb.write_text(
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
        "<permissions>&lol;</permissions>",
        encoding="utf-8",
    )

    config = parse_config_inputs(artifacts)

    assert config.files_failed == ("system/system/etc/permissions/bomb.xml",)
    assert config.files_read == 2
    assert config.privapp_permissions  # the other files still contributed


def test_an_oversized_config_xml_is_refused_unparsed(tmp_path, monkeypatch):
    import uadclaw.etcconfig as etcconfig_module

    monkeypatch.setattr(etcconfig_module, "MAX_CONFIG_XML_BYTES", 16)
    artifacts = build_scratch(tmp_path) / "artifacts"

    config = parse_config_inputs(artifacts)

    assert len(config.files_failed) == 2
    assert config.privapp_permissions == {}


def test_a_malformed_config_xml_is_named_rather_than_read_as_an_empty_allowlist(tmp_path):
    artifacts = build_scratch(tmp_path) / "artifacts"
    broken = artifacts / "system/system/etc/permissions/broken.xml"
    broken.write_text("<permissions><privapp-permissions package=", encoding="utf-8")

    config = parse_config_inputs(artifacts)

    assert config.files_failed == ("system/system/etc/permissions/broken.xml",)
    assert config.privapp_permissions


def test_merging_two_devices_config_inputs_unions_and_never_withdraws():
    left = ConfigInputs(
        privapp_permissions={"com.a": ("p1",)},
        static_role_holders={"role": ("com.a",)},
        platform_libraries=frozenset({"lib.a"}),
        files_read=2,
    )
    right = ConfigInputs(
        privapp_permissions={"com.a": ("p2",), "com.b": ("p3",)},
        platform_libraries=frozenset({"lib.b"}),
        files_read=3,
    )

    merged = merge_config_inputs([left, right])
    reversed_merge = merge_config_inputs([right, left])

    assert merged.privapp_permissions == {"com.a": ("p1", "p2"), "com.b": ("p3",)}
    assert merged.static_role_holders == {"role": ("com.a",)}
    assert merged.platform_libraries == frozenset({"lib.a", "lib.b"})
    assert merged.files_read == 5
    assert merged.as_json() == reversed_merge.as_json()


def test_config_inputs_survive_a_json_round_trip(tmp_path):
    config = parse_config_inputs(build_scratch(tmp_path) / "artifacts")

    assert ConfigInputs.from_json(config.as_json()).as_json() == config.as_json()


# --- the three stages ---------------------------------------------------------------------------


async def test_the_graph_stage_writes_both_edge_classes_and_the_evidence(
    db_env, db_session_factory, tmp_path
):
    ctx = await context(db_session_factory, tmp_path)
    await seed_corpus(db_session_factory, ctx.job_id)

    await corpus_graph_stage(ctx)

    rows = await analysis(db_session_factory)
    assert set(rows) == {item.package for item in CORPUS}
    assert rows["com.android.settings.overlay.oriole"].dependencies == ["com.android.settings"]
    assert rows["com.android.settings"].needed_by == ["com.android.settings.overlay.oriole"]
    assert rows["com.vendor.app"].dependencies == ["com.vendor.lib"]
    assert rows["com.vendor.lib"].needed_by == ["com.vendor.app"]
    assert rows["com.listed.one"].dependencies == []
    assert [edge["kind"] for edge in rows["com.vendor.app"].edges] == ["library"]
    assert rows["com.listed.one"].evidence["queries_packages_absent"] == []
    # A dex `content://` reference to another package's declared authority: evidence on the
    # card, zero edges on the graph — the same contract as the package queries.
    assert rows["com.example.contentclient"].dependencies == []
    assert rows["com.example.contentclient"].evidence["content_uri_authorities_in_corpus"] == [
        "com.example.content"
    ]
    assert rows["com.example.contentclient"].evidence["content_uri_authorities_absent"] == [
        "com.example.gone"
    ]

    # This stage is also the one that parks the `/etc` inputs, because it is the last one that
    # can still see the job's scratch and the ladder two stages later cannot.
    async with db_session_factory() as session:
        scan = (await session.execute(select(DeviceScan))).scalar_one()
    assert scan.config_inputs["privapp_permissions"] == {
        "com.example.allowlisted": ["android.permission.MANAGE_USERS", "android.permission.REBOOT"]
    }
    assert scan.config_inputs["static_role_holders"] == {
        "android.app.role.SMS": ["com.example.messages"]
    }
    assert scan.config_inputs["platform_libraries"] == ["android.test.base"]


async def test_the_graph_stage_refuses_a_job_whose_config_inputs_are_gone(
    db_env, db_session_factory, tmp_path
):
    ctx = StageContext(
        job_id=await seed_job(db_session_factory),
        attempt=1,
        scratch_dir=tmp_path / "empty-scratch",
        session_factory=db_session_factory,
    )
    await seed_corpus(db_session_factory, ctx.job_id)

    with pytest.raises(StageInputError, match="has no .*artifacts directory"):
        await corpus_graph_stage(ctx)


async def test_the_filter_stage_records_the_verdict_and_the_bytes_that_decided_it(
    db_env, db_session_factory, tmp_path, upstream_file
):
    ctx = await context(db_session_factory, tmp_path)
    await seed_corpus(db_session_factory, ctx.job_id)

    await filter_stage(ctx)

    rows = await analysis(db_session_factory)
    assert rows["com.listed.one"].queued is False
    assert rows["com.listed.one"].upstream_present is True
    assert rows["com.listed.one"].filter_verdict == "already_upstream"
    assert rows["com.vendor.app"].queued is True
    assert rows["com.vendor.app"].filter_verdict == "queued"
    provenance = rows["com.vendor.app"].upstream_provenance
    assert provenance["path"] == str(upstream_file)
    assert provenance["entry_count"] == 2
    assert len(provenance["sha256"]) == 64


async def test_the_filter_stage_refuses_to_run_without_the_upstream_list(
    db_env, db_session_factory, tmp_path, monkeypatch
):
    """Nothing is written: a run that treated the list as absent would mark the whole corpus
    as new, and the rows it wrote would be indistinguishable from a real result."""
    monkeypatch.setenv("UPSTREAM_LIST_PATH", str(tmp_path / "absent.json"))
    ctx = await context(db_session_factory, tmp_path)
    await seed_corpus(db_session_factory, ctx.job_id)

    with pytest.raises(UpstreamListError, match="is not a file"):
        await filter_stage(ctx)

    assert await analysis(db_session_factory) == {}


async def test_the_ladder_stage_writes_a_floor_from_inputs_no_manifest_carries(
    db_env, db_session_factory, tmp_path
):
    """Database to database: the `/etc` inputs were parked on the scan row a stage earlier,
    which is what lets a floor outlive the scratch directory its evidence came from."""
    ctx = await context(db_session_factory, tmp_path)
    await seed_corpus(db_session_factory, ctx.job_id)

    await corpus_graph_stage(ctx)
    await rule_ladder_stage(ctx)

    rows = await analysis(db_session_factory)
    assert rows["com.android.settings"].floor == Removal.UNSAFE
    assert rows["com.android.settings"].floor_rule == FloorRule.CORE_APP
    assert rows["com.android.settings.overlay.oriole"].floor == Removal.ADVANCED
    assert rows["com.vendor.lib"].floor == Removal.UNSAFE
    assert rows["com.vendor.lib"].floor_rule == FloorRule.SOLE_LIBRARY_PROVIDER
    # Sole holder of a static role, which `roles.xml` states and no manifest does. It also
    # happens to be the corpus's only SMS_DELIVER handler, so the strictest of the two wins.
    assert rows["com.example.messages"].floor == Removal.UNSAFE
    assert rows["com.example.messages"].floor_rule == FloorRule.SOLE_STATIC_ROLE_HOLDER
    assert [reason["rule"] for reason in rows["com.example.messages"].floor_reasons] == [
        FloorRule.SOLE_STATIC_ROLE_HOLDER,
        FloorRule.SOLE_CORE_INTENT_HANDLER,
        FloorRule.DEFAULT,
    ]
    assert rows["com.listed.one"].floor == Removal.RECOMMENDED
    # The allowlist entry came out of the XML, not out of any manifest.
    assert rows["com.example.allowlisted"].floor == Removal.ADVANCED
    assert rows["com.example.allowlisted"].privapp_allowlisted is True
    assert rows["com.example.allowlisted"].privapp_permission_count == 2


async def test_a_static_role_recorded_by_one_scan_reaches_the_whole_corpus(
    db_env, db_session_factory, tmp_path
):
    """The union is what makes the ladder corpus-wide: device B's run must see the role device
    A recorded, whose scratch is long gone."""
    ctx = await context(db_session_factory, tmp_path)
    await seed_corpus(db_session_factory, ctx.job_id)
    await corpus_graph_stage(ctx)

    async with db_session_factory() as session:
        config = await load_config_inputs(session)

    assert config.static_role_holders == {"android.app.role.SMS": ("com.example.messages",)}
    assert config.holders_of_static_role("com.example.messages") == ("android.app.role.SMS",)


async def test_every_stage_refuses_an_empty_corpus(db_env, db_session_factory, tmp_path):
    """A graph over nothing, a filter over nothing and a ladder over nothing all succeed
    silently, and the job goes green having analysed zero packages."""
    async with db_session_factory() as session:
        with pytest.raises(CorpusStoreError, match="package_facts is empty"):
            await require_corpus(session)


async def test_the_three_stages_do_not_blank_each_others_columns(
    db_env, db_session_factory, tmp_path, upstream_file
):
    """The wiring failure this file exists for: each stage upserts the same primary key, so
    one that named every column would wipe the other two on every run."""
    ctx = await context(db_session_factory, tmp_path)
    await seed_corpus(db_session_factory, ctx.job_id)

    await corpus_graph_stage(ctx)
    await filter_stage(ctx)
    await rule_ladder_stage(ctx)

    row = (await analysis(db_session_factory))["com.android.settings.overlay.oriole"]
    assert row.dependencies == ["com.android.settings"]
    assert row.queued is True
    assert row.floor == Removal.ADVANCED
    assert row.upstream_provenance is not None
    assert row.floor_reasons


async def test_running_the_whole_core_twice_produces_identical_rows(
    db_env, db_session_factory, tmp_path, upstream_file
):
    """The reproducibility property, at the row level: the deterministic core is the half of
    this pipeline that is allowed to claim it."""
    ctx = await context(db_session_factory, tmp_path)
    await seed_corpus(db_session_factory, ctx.job_id)
    snapshots = []

    for _ in range(2):
        await corpus_graph_stage(ctx)
        await filter_stage(ctx)
        await rule_ladder_stage(ctx)
        rows = await analysis(db_session_factory)
        snapshots.append(
            {
                package: {
                    "dependencies": row.dependencies,
                    "needed_by": row.needed_by,
                    "edges": row.edges,
                    "evidence": row.evidence,
                    "queued": row.queued,
                    "filter_verdict": row.filter_verdict,
                    "floor": row.floor,
                    "floor_rule": row.floor_rule,
                    "floor_reasons": row.floor_reasons,
                }
                for package, row in rows.items()
            }
        )

    assert len(snapshots[0]) == len(CORPUS)
    assert snapshots[0] == snapshots[1]


async def test_a_second_device_creates_edges_for_packages_an_earlier_job_already_analysed(
    db_env, db_session_factory, tmp_path, upstream_file
):
    """Edges are statements about what else exists, which is why every job re-runs the whole
    corpus rather than only its own device."""
    ctx = await context(db_session_factory, tmp_path)
    async with db_session_factory() as session, session.begin():
        await record_device_scan(
            session,
            job_id=ctx.job_id,
            device_key=PIXEL,
            build="b1",
            scanned_at=NOW,
            apk_total=1,
            parsed_ok=1,
            failures=[],
        )
        await store_device_facts(
            session,
            device_key=PIXEL,
            build="b1",
            facts=[
                make_facts("com.vendor.lib", libraries=(LibraryFact(name="com.vendor.support"),))
            ],
            observed_at=NOW,
        )
    await corpus_graph_stage(ctx)
    assert (await analysis(db_session_factory))["com.vendor.lib"].needed_by == []

    async with db_session_factory() as session, session.begin():
        await store_device_facts(
            session,
            device_key=EMULATOR,
            build="b2",
            facts=[make_facts("com.vendor.app", uses_libraries_required=("com.vendor.support",))],
            observed_at=NOW,
        )
    await corpus_graph_stage(ctx)

    assert (await analysis(db_session_factory))["com.vendor.lib"].needed_by == ["com.vendor.app"]
