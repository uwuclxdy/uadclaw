"""One approved package, emitted end to end onto a REAL clone of the upstream repo.

Every git test so far — `test_branch_stage.py` and `test_upstream_repo.py` — drives the stage
against a `git init` throwaway carrying a two-entry list, and the byte-level splice pins in
`test_emission.py` never run the STAGE against the real 1.6 MB file. This drives
`stages.branch_stage` — the worker's and the dashboard's path — over the real 393-package
corpus against a fresh clone of the real upstream repo, with one package approved the way
triage approves one.

    UADCLAW_HEAVY_TESTS=1 \\
      UADCLAW_HEAVY_APK_DIR=/path/to/oriole/artifacts \\
      UADCLAW_HEAVY_EMULATOR_APK_DIR=/path/to/emulator/artifacts \\
      UADCLAW_HEAVY_UPSTREAM_LIST=/path/to/uad_lists.json \\
      UADCLAW_HEAVY_UPSTREAM_CLONE=/path/to/a/clone/of/universal-android-debloater-ng \\
      uv run pytest -n0 -m heavy tests/test_emission_real_clone_heavy.py

`UADCLAW_HEAVY_UPSTREAM_CLONE` is never committed into: the test clones it fresh into its own
tmp dir, so a rerun starts clean whatever the last one left behind. The no-push half of the
verify line is pinned the way `test_branch_stage.py::test_nothing_is_pushed_fetched_or_
authenticated` pins it — that test's premise, a clone with no remote, cannot hold for a clone
of a real repo — so here the prepared clone's refs and the fresh clone's remote refs are
snapshotted before the stage and must not move.

Every count below was measured on 2026-08-13 against the real corpora and is asserted exactly.
"""

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from uadclaw import jobs as jobs_module
from uadclaw.bundle import build_bundles
from uadclaw.classify import Classification, Confidence, UadList
from uadclaw.classifystore import load_identities, store_classification
from uadclaw.corpus import build_graph
from uadclaw.corpusstore import (
    load_config_inputs,
    load_corpus,
    record_config_inputs,
    store_filter_verdicts,
    store_floors,
    store_graph,
)
from uadclaw.emission import DOMINANT_KEY_ORDER, branch_name, build_entry, vendor_for
from uadclaw.emissionstore import load_approved
from uadclaw.etcconfig import parse_config_inputs
from uadclaw.facts import ApkFacts
from uadclaw.factstore import record_device_scan, store_device_facts
from uadclaw.filters import FilterVerdict, queue_verdicts
from uadclaw.ladder import compute_floors
from uadclaw.models import BranchEmission, JobKind, PackageAnalysis, PackageFact
from uadclaw.settings import get_settings
from uadclaw.stages import apk_paths, branch_stage, parse_device_apks
from uadclaw.triagestore import record_decision
from uadclaw.upstream import load_upstream_list
from uadclaw.worker import StageContext

REPO_ROOT = Path(__file__).resolve().parents[1]

ORIOLE_ARTIFACTS = os.environ.get("UADCLAW_HEAVY_APK_DIR", "")
EMULATOR_ARTIFACTS = os.environ.get("UADCLAW_HEAVY_EMULATOR_APK_DIR", "")
UPSTREAM_LIST = os.environ.get("UADCLAW_HEAVY_UPSTREAM_LIST", "")

ORIOLE_DEVICE = "pixel:oriole"
ORIOLE_BUILD = "cp2a.260705.006.a1"
EMULATOR_DEVICE = "google:emulator-a16"
EMULATOR_BUILD = "android-36.1-google_apis-x86_64"

VENDOR = "pixel"
MODEL = "deepseek-v4-flash"
LIST_PATH = "resources/assets/uad_lists.json"

# The candidate under test: the lexicographically-first queued package whose device
# provenance names only the pixel driver. Deterministic from the real corpora, so a corpus
# change that moves it fails this test loudly instead of silently emitting a different
# package. Measured on the 2026-08-13 corpora.
CHOSEN_PACKAGE = "com.android.appsearch.aiseal.config"

# The disclosure names the commit that produced the entries. The deploy reads this value from
# the deploying checkout, so the test reads it from its own rather than inventing one.
PIPELINE_SHA = subprocess.run(  # noqa: S603
    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
    capture_output=True,
    check=True,
    text=True,
    timeout=30,
).stdout.strip()

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_HEAVY_TESTS") != "1",
        reason="opt-in: set UADCLAW_HEAVY_TESTS=1 (needs both extracted corpora and a clone)",
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


def git(root: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args], capture_output=True, check=True, text=True, timeout=120
    )
    return result.stdout.strip()


def git_bytes(root: Path, spec: str) -> bytes:
    """A committed blob's exact bytes, via `cat-file`. Read as BYTES: the digest comparisons
    below are over the bytes, and the text helper strips trailing newlines."""
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "cat-file", "blob", spec],
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout


def git_diff(root: Path, base: str, head: str, path: str) -> str:
    """`git diff` between two differing commits exits 1, which is the interesting case."""
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "diff", base, head, "--", path],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.stdout


def all_refs(root: Path) -> str:
    """Every ref and the object it names, one line each: a push or a fetch moves one of these."""
    return git(root, "for-each-ref", "--format=%(refname) %(objectname)")


def _duplicated_keys(raw: bytes) -> frozenset[str]:
    """The keys the base document already carries twice WITHIN ONE OBJECT.

    "No duplicate keys" is not a property the destination file has: the real file's
    top-level object carries three package names twice (`com.aura.oobe.motorola`,
    `com.dti.motorola`, `com.aura.oobe.kddi`, measured 2026-08-13). What the emission must
    keep is "no duplicate key BEYOND those", which is what `_no_new_duplicates` refuses.

    The seen-set is per OBJECT, not per document: a field name like `description` occurs in
    every entry, and a document-global set would read the hundredth occurrence as a
    duplicate, polluting the allowed set with every common field name.
    """
    duplicated: set[str] = set()

    def collect(pairs):
        obj_seen: set[str] = set()
        for key, _ in pairs:
            if key in obj_seen:
                duplicated.add(key)
            else:
                obj_seen.add(key)
        return dict(pairs)

    json.loads(raw, object_pairs_hook=collect)
    return frozenset(duplicated)


def _no_new_duplicates(allowed: frozenset[str]):
    """A json `object_pairs_hook` that refuses any duplicate key the base does not carry.

    A plain `json.loads` collapses a repeated key silently — the duplicate-key document
    `insert_entries`'s own docstring warns upstream's serde round-trip does not catch — so
    every assertion reading the parsed dict would pass over a splice that appended the batch
    entry twice. Applied to every object in the emitted document, not only the top level,
    and the seen-set is per OBJECT for the reason `_duplicated_keys` states: with a
    document-global set, the allowed set absorbs every common field name and a doubled field
    inside one entry slips through because an earlier entry already saw it once.
    """

    def hook(pairs):
        result = {}
        obj_seen: set[str] = set()
        for key, value in pairs:
            if key in obj_seen and key not in allowed:
                raise AssertionError(f"the emission added a duplicate key {key!r}")
            obj_seen.add(key)
            result[key] = value
        return result

    return hook


def prepared_clone() -> Path:
    """The prepared clone, read-only. Set-but-missing FAILS rather than skipping, so a run
    that asks for the heavy input by name cannot silently not use it."""
    value = os.environ.get("UADCLAW_HEAVY_UPSTREAM_CLONE", "")
    if not value:
        pytest.skip("set UADCLAW_HEAVY_UPSTREAM_CLONE to a local clone of the upstream repo")
    path = Path(value)
    assert path.is_dir(), f"UADCLAW_HEAVY_UPSTREAM_CLONE is not a directory: {path}"
    return path


@pytest.fixture(autouse=True)
def _isolate_git_config(monkeypatch):
    """Same isolation as `test_branch_stage`: git subprocesses this test runs inherit the
    environment, and this box's global config points every commit at the commit-message
    linter. The stage's own git calls neutralise config themselves; this covers the test's."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


@pytest.fixture
def clone(tmp_path) -> tuple[Path, str]:
    """A fresh local clone of the prepared upstream clone, plus that clone's default branch.

    Cloning fresh per run is what makes reruns idempotent: the emission commits into THIS
    clone and the prepared one is never touched. The base ref is read off the prepared
    clone's own `origin/HEAD` rather than assumed to be `main`, and the emission is then
    configured with exactly that ref.
    """
    source = prepared_clone()
    base_ref = git(source, "symbolic-ref", "refs/remotes/origin/HEAD").removeprefix(
        "refs/remotes/origin/"
    )
    assert base_ref
    target = tmp_path / "uad-clone"
    subprocess.run(  # noqa: S603
        ["git", "clone", "--quiet", str(source), str(target)],
        capture_output=True,
        check=True,
        timeout=300,
    )
    # The emission's commit needs an identity, and nothing upstream of it supplies one: this
    # box's global config is neutralised for the clone's git calls, and the prepared clone
    # carries no `user.*` of its own. Same setup as test_branch_stage's throwaway clones.
    git(target, "config", "user.name", "Pipeline Test")
    git(target, "config", "user.email", "pipeline@example.invalid")
    return target, base_ref


@pytest.fixture
def emission_env(monkeypatch, clone):
    """Point the branch stage at the fresh clone, the real list path and the clone's actual
    default branch, and name the pipeline commit the disclosure must carry."""
    target, base_ref = clone
    monkeypatch.setenv("UPSTREAM_REPO_PATH", str(target))
    monkeypatch.setenv("UPSTREAM_REPO_LIST_PATH", LIST_PATH)
    monkeypatch.setenv("UPSTREAM_BASE_REF", base_ref)
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", PIPELINE_SHA)
    get_settings.cache_clear()
    return target, base_ref


async def store(session_factory, device: str, build: str, facts: list[ApkFacts]) -> None:
    async with session_factory() as session, session.begin():
        await store_device_facts(
            session, device_key=device, build=build, facts=facts, observed_at=datetime.now(UTC)
        )


async def test_one_approved_package_emits_a_branch_on_a_real_upstream_clone(
    db_env, db_session_factory, oriole_facts, emulator_facts, upstream, emission_env
):
    """The verify line: one approved package over the real 393-package corpus, loaded by
    `load_approved` and committed onto a branch of a real clone of the real repo, with the
    real 1.6 MB file's bytes surviving byte for byte."""
    clone, base_ref = emission_env
    at = datetime.now(UTC)

    # The stages read the upstream list from the setting, not from this test's env var: the
    # deployed worker filters and anchors against UPSTREAM_LIST_PATH. When a file sits at the
    # configured path, it must be the same bytes this test derives every verdict and bundle
    # anchor from, or a green run pins a different file than production reads.
    configured_list = get_settings().upstream_list_path
    if configured_list.is_file():
        assert configured_list.read_bytes() == Path(UPSTREAM_LIST).read_bytes(), (
            f"{configured_list} differs from UADCLAW_HEAVY_UPSTREAM_LIST; refresh the test "
            "input to match what the stages would read"
        )

    # --- the real corpus, through the real writers --------------------------------------
    await store(db_session_factory, ORIOLE_DEVICE, ORIOLE_BUILD, oriole_facts)
    await store(db_session_factory, EMULATOR_DEVICE, EMULATOR_BUILD, emulator_facts)

    # The /etc inputs travel through device_scans, exactly as `corpus_graph` parks them: the
    # bundle below has to be derivable from the database alone, which is what the
    # classification stage derives it from.
    async with db_session_factory() as session, session.begin():
        oriole_job = await jobs_module.create_job(
            session,
            kind=JobKind.FIRMWARE_ANALYSIS.value,
            params={"driver": "pixel", "device": "oriole"},
        )
        emulator_job = await jobs_module.create_job(
            session,
            kind=JobKind.FIRMWARE_ANALYSIS.value,
            params={"driver": "google", "device": "emulator-a16"},
        )
    async with db_session_factory() as session, session.begin():
        await record_device_scan(
            session,
            job_id=oriole_job.id,
            device_key=ORIOLE_DEVICE,
            build=ORIOLE_BUILD,
            scanned_at=at,
            apk_total=len(oriole_facts),
            parsed_ok=len(oriole_facts),
            failures=[],
        )
        await record_device_scan(
            session,
            job_id=emulator_job.id,
            device_key=EMULATOR_DEVICE,
            build=EMULATOR_BUILD,
            scanned_at=at,
            apk_total=len(emulator_facts),
            parsed_ok=len(emulator_facts),
            failures=[],
        )
    async with db_session_factory() as session, session.begin():
        await record_config_inputs(
            session, job_id=oriole_job.id, config=parse_config_inputs(Path(ORIOLE_ARTIFACTS))
        )
        await record_config_inputs(
            session, job_id=emulator_job.id, config=parse_config_inputs(Path(EMULATOR_ARTIFACTS))
        )

    # --- the deterministic core, the way test_the_whole_core_writes_one_analysis_row_per_package
    # runs it: facts -> corpus graph -> filter -> rule ladder, every floor computed.
    async with db_session_factory() as session, session.begin():
        corpus = await load_corpus(session)
        config = await load_config_inputs(session)
        verdicts = queue_verdicts(corpus, upstream=upstream)
        floors = compute_floors(corpus, config=config)
        await store_graph(
            session,
            corpus,
            build_graph(corpus, platform_libraries=config.platform_libraries),
            at=at,
        )
        await store_filter_verdicts(session, verdicts, upstream=upstream, at=at)
        await store_floors(session, floors, at=at)

    async with db_session_factory() as session:
        fact_rows = (
            await session.execute(select(func.count()).select_from(PackageFact))
        ).scalar_one()
        analysis = (await session.execute(PackageAnalysis.__table__.select())).mappings().all()
    assert fact_rows == 393
    assert len(analysis) == 393
    assert all(row["floor"] is not None for row in analysis)
    assert sum(1 for row in analysis if row["queued"]) == 48
    assert upstream.entry_count == 5372

    # --- one deterministic candidate, approved the way triage approves one --------------
    by_name = {item.package: item for item in corpus}
    candidates = sorted(
        name
        for name, verdict in verdicts.items()
        if verdict is FilterVerdict.QUEUED and vendor_for(by_name[name].devices) == VENDOR
    )
    assert candidates
    chosen = candidates[0]
    assert chosen == CHOSEN_PACKAGE
    floor = floors[chosen]
    assert str(floor.floor) == "Recommended"

    # The bundle the way `llm_stage` builds it: derived from the database alone, because that
    # is the derivation the classification stage promises to be reproducible from.
    async with db_session_factory() as session:
        identities = await load_identities(session)
    graph = build_graph(corpus, platform_libraries=config.platform_libraries)
    bundles = build_bundles(
        corpus, floors=floors, identities=identities, graph=graph, upstream=upstream
    )
    bundle = bundles[chosen]

    stored = next(row for row in analysis if row["package"] == chosen)
    assert stored["filter_verdict"] == "queued"
    assert stored["floor"] == str(floor.floor)

    async with db_session_factory() as session, session.begin():
        await store_classification(
            session,
            Classification(
                package=chosen,
                bundle_sha256=bundle.sha256,
                description=f"Vendor application {chosen}. Removing it loses its local data.",
                list=UadList.MISC,
                removal=floor.floor,
                confidence=Confidence.MEDIUM,
                unknown_fields=(),
                reasoning_brief="No privileged surface.",
                provenance={"description": f"llm:{MODEL}", "removal": f"llm:{MODEL}"},
            ),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=at,
        )
    async with db_session_factory() as session, session.begin():
        await record_decision(
            session, package=chosen, bundle_sha256=bundle.sha256, action="approve", at=at
        )

    # --- load_approved over the full 393-row corpus -------------------------------------
    async with db_session_factory() as session:
        approved = await load_approved(session, vendor=VENDOR)
    async with db_session_factory() as session:
        assert await load_approved(session, vendor="google") == ()
    assert [item.package for item in approved] == [chosen]
    assert approved[0].bundle_sha256 == bundle.sha256
    assert approved[0].removal == str(floor.floor)
    assert approved[0].floor == str(floor.floor)

    # --- the stage, driven exactly as the worker drives it ------------------------------
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.BRANCH_EMISSION.value, params={"vendor": VENDOR}
        )
    ctx = StageContext(
        job_id=job.id, attempt=1, scratch_dir=None, session_factory=db_session_factory
    )

    source = prepared_clone()
    source_refs_before = all_refs(source)
    remote_refs_before = git(
        clone, "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/"
    )
    await branch_stage(ctx)

    # --- the branch, the row, and the emitted bytes -------------------------------------
    branch = branch_name(
        prefix=get_settings().emission_branch_prefix, vendor=VENDOR, job_id=str(job.id)
    )
    committed = git(clone, "rev-parse", f"refs/heads/{branch}")
    base_commit = git(clone, "rev-parse", "refs/remotes/origin/HEAD")
    async with db_session_factory() as session:
        row = (await session.execute(select(BranchEmission))).scalar_one()
    assert row.commit_oid == committed
    assert row.base_commit == base_commit
    assert row.package_count == 1
    assert row.vendor == VENDOR
    assert row.reconciled is False
    assert PIPELINE_SHA in row.pr_body
    assert MODEL in row.pr_body
    assert bundle.sha256 in row.pr_body

    emitted = git_bytes(clone, f"{committed}:{LIST_PATH}")
    assert hashlib.sha256(emitted).hexdigest() == row.list_sha256
    # The clone is left ON the branch for the human to push, carrying the committed bytes.
    assert git(clone, "symbolic-ref", "--short", "HEAD") == branch
    assert (clone / LIST_PATH).read_bytes() == emitted

    # The emitted file under the upstream schema the pipeline's own loader implements: it
    # parses, every pre-existing entry survived the append unchanged, and the batch entry is
    # present in the dominant key order with exactly the approved fields. The parse refuses
    # any duplicate key the base does not already carry, because json's ordinary parse
    # collapses a repeated key and would pass every assertion here over a splice that
    # appended the batch entry twice.
    base_bytes = git_bytes(clone, f"{base_commit}:{LIST_PATH}")
    before = json.loads(base_bytes)
    parsed = json.loads(emitted, object_pairs_hook=_no_new_duplicates(_duplicated_keys(base_bytes)))
    assert len(parsed) == upstream.entry_count + 1
    assert all(parsed[name] == value for name, value in before.items())
    assert set(parsed) - set(before) == {chosen}
    assert list(parsed[chosen]) == list(DOMINANT_KEY_ORDER)
    assert parsed[chosen] == build_entry(approved[0])
    list_copy = clone.parent / "emitted-uad_lists.json"
    list_copy.write_bytes(emitted)
    reloaded = load_upstream_list(list_copy)
    assert reloaded.entry_count == upstream.entry_count + 1
    assert chosen in reloaded.packages

    # Append-only, pinned against the REAL file the splice has only ever met in test_emission.
    # The diff is asserted non-empty before any no-matches result is trusted: a moved or
    # reverted clone would otherwise answer "no removed lines" vacuously.
    diff = git_diff(clone, base_commit, committed, LIST_PATH)
    assert diff, "the diff is empty: the batch never reached the file"
    changed = [
        line
        for line in diff.splitlines()
        if line.startswith(("-", "+")) and not line.startswith(("---", "+++"))
    ]
    removed = [line for line in changed if line.startswith("-")]
    added = [line for line in changed if line.startswith("+")]
    assert added
    assert any(chosen in line for line in added)
    assert removed == []
    # The commit touches only the list.
    assert git(clone, "diff-tree", "--no-commit-id", "--name-only", "-r", committed) == LIST_PATH

    # Nothing pushed, fetched or authenticated (the no-push premise from
    # test_nothing_is_pushed_fetched_or_authenticated, restated for a clone that HAS a
    # remote): no ref anywhere moved.
    assert all_refs(source) == source_refs_before
    assert (
        git(clone, "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/")
        == remote_refs_before
    )

    # A green run leaves nothing behind: pytest's default tmp_path retention keeps every
    # run's ~11 MB clone forever. A FAILED run keeps its clone — that is the one worth
    # reading — because the cleanup only runs past every assertion above.
    list_copy.unlink()
    shutil.rmtree(clone)
