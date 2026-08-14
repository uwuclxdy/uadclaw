"""The deterministic core against both real corpora, end to end and opt-in.

The fast tests own the logic. What only real firmware answers is how much the two edge classes
actually yield and how far the filter really collapses the 15:1 junk ratio, and both answers
are counts that no synthetic corpus can be honest about: an author who builds the fixture
chooses the overlap.

    UADCLAW_HEAVY_TESTS=1 \\
      UADCLAW_HEAVY_APK_DIR=/path/to/oriole/artifacts \\
      UADCLAW_HEAVY_EMULATOR_APK_DIR=/path/to/emulator/artifacts \\
      UADCLAW_HEAVY_UPSTREAM_LIST=/path/to/uad_lists.json \\
      uv run pytest -n0 -m heavy tests/test_corpus_heavy.py

Both APK directories are `<partition>/<path-inside-the-image>` trees as
`unpack.extract_artifacts` writes them, complete with the `/etc` config XMLs, so nothing here
unpacks an image. Every count was measured on 2026-08-11 and is asserted exactly: a graph that
silently stops emitting edges returns zero, which looks exactly like a corpus that has none.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from uadclaw.corpus import EDGE_LIBRARY, EDGE_OVERLAY, build_graph
from uadclaw.corpusstore import load_corpus, store_filter_verdicts, store_floors, store_graph
from uadclaw.etcconfig import parse_config_inputs
from uadclaw.facts import ApkFacts
from uadclaw.factstore import store_device_facts
from uadclaw.filters import FilterVerdict, queue_verdicts, survival
from uadclaw.ladder import FloorRule, Removal, compute_floors
from uadclaw.models import PackageAnalysis
from uadclaw.stages import apk_paths, parse_device_apks
from uadclaw.upstream import load_upstream_list

ORIOLE_ARTIFACTS = os.environ.get("UADCLAW_HEAVY_APK_DIR", "")
EMULATOR_ARTIFACTS = os.environ.get("UADCLAW_HEAVY_EMULATOR_APK_DIR", "")
UPSTREAM_LIST = os.environ.get("UADCLAW_HEAVY_UPSTREAM_LIST", "")

ORIOLE_DEVICE = "pixel:oriole"
ORIOLE_BUILD = "cp2a.260705.006.a1"
EMULATOR_DEVICE = "google:emulator-a16"
EMULATOR_BUILD = "android-36.1-google_apis-x86_64"

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_HEAVY_TESTS") != "1",
        reason="opt-in: set UADCLAW_HEAVY_TESTS=1 (needs both extracted corpora)",
    ),
]


def facts_from(directory: str, label: str) -> list[ApkFacts]:
    if not directory:
        pytest.skip(f"set {label} to an extracted <partition>/<path> tree")
    root = Path(directory)
    if not root.is_dir():
        pytest.skip(f"{label} is not a directory: {root}")
    parsed, failures = parse_device_apks(root, apk_paths(root))
    assert failures == [], f"{label} parsed with zero failures when this baseline was measured"
    return parsed


@pytest.fixture(scope="module")
def oriole_facts() -> list[ApkFacts]:
    return facts_from(ORIOLE_ARTIFACTS, "UADCLAW_HEAVY_APK_DIR")


@pytest.fixture(scope="module")
def emulator_facts() -> list[ApkFacts]:
    return facts_from(EMULATOR_ARTIFACTS, "UADCLAW_HEAVY_EMULATOR_APK_DIR")


@pytest.fixture(scope="module")
def upstream():
    if not UPSTREAM_LIST:
        pytest.skip("set UADCLAW_HEAVY_UPSTREAM_LIST to a copy of the upstream uad_lists.json")
    return load_upstream_list(Path(UPSTREAM_LIST))


async def store(session_factory, device: str, build: str, facts: list[ApkFacts]) -> None:
    async with session_factory() as session, session.begin():
        await store_device_facts(
            session, device_key=device, build=build, facts=facts, observed_at=datetime.now(UTC)
        )


# --- the corpus graph, on real firmware ---------------------------------------------------


async def test_the_overlay_class_carries_the_graph_and_the_library_class_yields_nothing(
    db_env, db_session_factory, oriole_facts, emulator_facts
):
    """The measured answer, and it is lopsided.

    Every edge on both corpora is an overlay. The library class yields **zero**, and that is
    the real data rather than a broken lookup: each image declares exactly one
    `<static-library>` (`com.google.android.trichromelibrary`) that nothing in the image
    consumes, and every `<uses-library required="true">` name resolves to a Java shared library
    the PLATFORM declares in `/etc/permissions/*.xml` — `android.test.base`,
    `com.google.android.dialer.support`, `org.apache.http.legacy` — which is not a removable
    package and therefore not an edge. Loosening the rule to close that gap would be inventing
    dependencies, so the number stays at zero and is asserted at zero.
    """
    await store(db_session_factory, ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts)
    async with db_session_factory() as session:
        oriole = await load_corpus(session)
    oriole_graph = build_graph(oriole)

    await store(db_session_factory, EMULATOR_DEVICE, EMULATOR_BUILD, emulator_facts)
    async with db_session_factory() as session:
        both = await load_corpus(session)
    union_graph = build_graph(both)

    assert len(oriole) == 312
    assert oriole_graph.counts_by_kind() == {EDGE_OVERLAY: 71, EDGE_LIBRARY: 0}
    assert len(both) == 393
    assert union_graph.counts_by_kind() == {EDGE_OVERLAY: 138, EDGE_LIBRARY: 0}
    # 156 overlay packages, 138 of them aimed at a target the corpus carries. The other 18 are
    # recorded as evidence rather than guessed at.
    assert sum(bool(item.overlay_target) for item in both) == 156
    assert (
        sum(
            1
            for evidence in union_graph.evidence.values()
            if evidence.overlay_target_absent is not None
        )
        == 18
    )


async def test_the_declared_package_queries_are_recorded_and_none_of_them_is_an_edge(
    db_env, db_session_factory, oriole_facts
):
    """510 declared queries across 65 Pixel packages, and not one of them may become a
    dependency: a caller is expected to handle the queried package being absent."""
    await store(db_session_factory, ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts)
    async with db_session_factory() as session:
        corpus = await load_corpus(session)
    graph = build_graph(corpus)

    declared = sum(len(item.queries_packages) for item in corpus)
    packages_declaring = sum(1 for item in corpus if item.queries_packages)
    recorded = sum(
        len(evidence.queries_packages_in_corpus) + len(evidence.queries_packages_absent)
        for evidence in graph.evidence.values()
    )

    assert (packages_declaring, declared) == (65, 510)
    assert recorded == declared
    assert {edge.kind for edge in graph.edges} == {EDGE_OVERLAY}


async def test_the_real_etc_trees_carry_the_ladder_inputs_no_apk_states(oriole_facts):
    """Measured on the real config trees. `roles.xml` is absent from both builds, which is
    normal and not an extraction failure, so the static-role clause has nothing to fire on
    here and says so rather than inventing a holder."""
    config = parse_config_inputs(Path(ORIOLE_ARTIFACTS))

    assert config.files_read == 246
    assert len(config.privapp_permissions) == 178
    assert config.privapp_allowlisted("com.google.android.gms")
    assert config.static_role_holders == {}
    assert "android.test.base" in config.platform_libraries
    assert len(config.platform_libraries) == 33
    assert config.files_failed == ()


# --- the filter, against the local probe's 15:1 --------------------------------------------


async def test_the_filter_survival_ratio_against_the_15_to_1_baseline(
    db_env, db_session_factory, oriole_facts, emulator_facts, upstream
):
    """`docs/research/local-probe.md` measured the noise floor by hand on the emulator image:
    228 packages, 152 already upstream, 76 missing, 40 of those emulator artifacts or
    auto-generated RRO, ~5 genuinely worth an entry — a 15:1 junk ratio on the RAW misses.

    Run mechanically over both corpora, the funnel lands on 48 survivors out of 393 packages.
    Two things are worth reading off that, and neither is "the filter fixed the ratio":

    - **Survival is the same order as the manual pass.** The probe's own post-filter figure
      was 36 of 228 (15.8%); this is 48 of 393 (12.2%), and every survivor is a Pixel package,
      so against the real device it is 48 of 312 (15.4%). The mechanical rules reproduce a
      hand pass done on a different image. (The emulator on its own queues nothing at all:
      every package it carries alone is emulator-only by definition. That is the rule working,
      and it is why the ratio is measured against the phone.)
    - **The 15:1 is not collapsed, and cannot be.** Every one of the three rules answers a
      question about provenance ("is it already carried", "is it generated", "is it only on an
      emulator"). None of them answers "is this worth an entry" — the survivors still include
      ~10 `com.google.android.gms.dynamite_*` modules and a pile of per-device settings
      overlays. That judgement is exactly what the model and the human gate are for, and the
      residual ratio is the load they carry.
    """
    await store(db_session_factory, ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts)
    await store(db_session_factory, EMULATOR_DEVICE, EMULATOR_BUILD, emulator_facts)
    async with db_session_factory() as session:
        corpus = await load_corpus(session)

    verdicts = queue_verdicts(corpus, upstream=upstream)
    counts = survival(verdicts)
    queued = {package for package, verdict in verdicts.items() if verdict is FilterVerdict.QUEUED}
    oriole_packages = {item.package for item in oriole_facts}

    assert upstream.entry_count == 5372
    assert counts == {
        "total": 393,
        "already_upstream": 273,
        "auto_generated_rro": 5,
        "emulator_only": 67,
        "queued": 48,
    }
    # Every survivor is a package a real phone ships: the emulator contributes none.
    assert queued <= oriole_packages
    # Four of the five the probe named by hand as genuinely worth an entry.
    assert {
        "com.android.privatespace",
        "com.google.android.trichromelibrary",
        "com.android.security.fsverity_metadata.system",
        "com.android.security.fsverity_metadata.system_ext",
    } <= queued
    # The fifth is the emulator-only rule's cost, stated rather than hidden: this Pixel build
    # does not ship `com.google.android.telephony.satellite` and the emulator does, so the
    # only image carrying it is an emulator and it drops. A second real device puts it back.
    assert verdicts["com.google.android.telephony.satellite"] is FilterVerdict.EMULATOR_ONLY
    # Overlays are not dropped as a class, and the survivors prove it rather than a comment.
    assert sum(1 for item in corpus if item.package in queued and item.overlay_target) == 17


async def test_an_auto_generated_rro_already_carried_upstream_reports_the_upstream_reason(
    db_env, db_session_factory, oriole_facts, upstream
):
    """26 of a Pixel's 27 auto-generated RROs are already in the list. A drop reason that
    reported the RRO rule instead would misdescribe how much of the corpus upstream covers."""
    await store(db_session_factory, ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts)
    async with db_session_factory() as session:
        corpus = await load_corpus(session)
    verdicts = queue_verdicts(corpus, upstream=upstream)

    rro = {item.package for item in corpus if "auto_generated_rro_" in item.package}

    assert len(rro) == 27
    assert sum(verdicts[package] is FilterVerdict.AUTO_GENERATED_RRO for package in rro) == 1
    assert sum(verdicts[package] is FilterVerdict.ALREADY_UPSTREAM for package in rro) == 26


# --- the ladder, on real firmware -----------------------------------------------------------


async def test_the_floor_distribution_over_both_real_corpora(
    db_env, db_session_factory, oriole_facts, emulator_facts
):
    """Pinned as a distribution rather than as spot checks: a rule that stops firing moves a
    whole band, and the band that must never shrink silently is `Unsafe`."""
    await store(db_session_factory, ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts)
    await store(db_session_factory, EMULATOR_DEVICE, EMULATOR_BUILD, emulator_facts)
    async with db_session_factory() as session:
        corpus = await load_corpus(session)
    config = parse_config_inputs(Path(ORIOLE_ARTIFACTS))

    floors = compute_floors(corpus, config=config)
    distribution: dict[str, int] = {}
    for floor in floors.values():
        distribution[str(floor.floor)] = distribution.get(str(floor.floor), 0) + 1
    by_rule: dict[str, int] = {}
    for floor in floors.values():
        by_rule[str(floor.rule)] = by_rule.get(str(floor.rule), 0) + 1

    assert distribution == {
        "Unsafe": 27,
        "Expert": 17,
        "Advanced": 265,
        "Recommended": 84,
    }
    # 23 `coreApp` packages plus the 4 the jurisdiction deny-list catches; none of the four is
    # also a `coreApp`, which is why the two numbers add to the `Unsafe` band exactly.
    assert by_rule[str(FloorRule.CORE_APP)] == 23
    assert by_rule[str(FloorRule.DENY_LIST)] == 4
    assert by_rule[str(FloorRule.SOLE_CORE_INTENT_HANDLER)] == 1


async def test_the_deny_list_catches_the_alerting_packages_both_images_ship(
    db_env, db_session_factory, oriole_facts, emulator_facts
):
    """A jurisdiction question no static analysis answers, so it is hardcoded — and these are
    the packages it exists for. `com.android.cellbroadcastreceiver.overlay.pixel` is the case
    that matters most: the overlay clause caps an inherited floor at `Advanced`, and it must
    not pull a deny-listed overlay down with it."""
    await store(db_session_factory, ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts)
    await store(db_session_factory, EMULATOR_DEVICE, EMULATOR_BUILD, emulator_facts)
    async with db_session_factory() as session:
        corpus = await load_corpus(session)

    floors = compute_floors(corpus)

    for package in (
        "com.android.cellbroadcastreceiver",
        "com.android.cellbroadcastreceiver.overlay.pixel",
        "com.android.cellbroadcastservice.overlay.pixel",
        "com.android.emergency",
    ):
        assert floors[package].floor == Removal.UNSAFE, package
    assert floors["com.android.cellbroadcastreceiver.overlay.pixel"].rule == FloorRule.DENY_LIST


async def test_the_whole_core_writes_one_analysis_row_per_package(
    db_env, db_session_factory, oriole_facts, emulator_facts, upstream
):
    """The three stages' outputs on one row, over the real corpus: 393 packages in, 393 rows
    out, every column populated."""
    await store(db_session_factory, ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts)
    await store(db_session_factory, EMULATOR_DEVICE, EMULATOR_BUILD, emulator_facts)
    at = datetime.now(UTC)

    async with db_session_factory() as session, session.begin():
        corpus = await load_corpus(session)
        config = parse_config_inputs(Path(ORIOLE_ARTIFACTS))
        await store_graph(
            session,
            corpus,
            build_graph(corpus, platform_libraries=config.platform_libraries),
            at=at,
        )
        await store_filter_verdicts(
            session, queue_verdicts(corpus, upstream=upstream), upstream=upstream, at=at
        )
        await store_floors(session, compute_floors(corpus, config=config), at=at)

    async with db_session_factory() as session:
        rows = (await session.execute(PackageAnalysis.__table__.select())).mappings().all()

    assert len(rows) == 393
    assert all(row["floor"] is not None for row in rows)
    assert all(row["queued"] is not None for row in rows)
    assert all(row["upstream_provenance"]["sha256"] == upstream.sha256 for row in rows)
    assert sum(row["queued"] for row in rows) == 48
    assert sum(bool(row["dependencies"]) for row in rows) == 138


async def test_the_content_uri_references_are_evidence_and_never_edges(
    db_env, db_session_factory, emulator_facts
):
    """The dex scan's output through the real store -> corpus -> graph path, measured
    2026-08-14 on the emulator corpus: 48 packages reference `content://` authorities,
    landing as 229 per-package evidence entries (82 in-corpus, 147 absent), and not one of
    them is an edge. The in-corpus half is real data rather than a broken lookup: every
    authority in it is declared by some other package on the same corpus. The absent half
    carries the runtime fragments (`%s`, `content:// scheme`) alongside genuine off-corpus
    providers, which is why the fuzzy match records authorities rather than resolutions.
    """
    await store(db_session_factory, EMULATOR_DEVICE, EMULATOR_BUILD, emulator_facts)
    async with db_session_factory() as session:
        corpus = await load_corpus(session)
    graph = build_graph(corpus)

    assert graph.counts_by_kind() == {EDGE_OVERLAY: 98, EDGE_LIBRARY: 0}
    assert sum(1 for item in corpus if item.content_uri_authorities) == 48
    declared: dict[str, set[str]] = {}
    for item in corpus:
        for authority in item.provider_authorities:
            declared.setdefault(authority, set()).add(item.package)
    in_corpus_total = 0
    absent_total = 0
    for item in corpus:
        evidence = graph.evidence[item.package]
        for authority in evidence.content_uri_authorities_in_corpus:
            assert declared.get(authority, set()) - {item.package}, (
                f"{item.package} has {authority} in-corpus but nobody else declares it"
            )
        in_corpus_total += len(evidence.content_uri_authorities_in_corpus)
        absent_total += len(evidence.content_uri_authorities_absent)
    assert in_corpus_total == 82
    assert absent_total == 147
