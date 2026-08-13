"""The branch stage end to end, against real throwaway clones and a real Postgres.

Nothing here mocks git. `upstreamrepo` is subprocess work whose whole value is that it knows
what git actually does — a plain checkout carries a staged edit, `check-ref-format --branch @`
echoes `@` back and exits 0 — so a stage test over a mocked one would pin this file's idea of
git rather than git. Every clone is built by `git init` under `tmp_path` and read back with
git, the same way `test_upstream_repo.py` does, and isolated from this box's own git
configuration so the global `core.hooksPath` commit-message linter does not run inside them.

The stage's hard part is not the happy path, it is the crash window: the emission row is
committed BEFORE the git commit it describes, so a worker killed between the two leaves a row
with no outcome and a branch nobody recorded. Four states a retry can meet, all four exercised
here, and the property under all of them is the same — a re-run neither cuts a second branch
nor wedges the job, and the row says plainly which of the two ways it reached its commit.
"""

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, text

from uadclaw import jobs as jobs_module
from uadclaw.classify import Classification, Confidence, UadList
from uadclaw.classifystore import store_classification
from uadclaw.emission import EmissionError, branch_name
from uadclaw.emissionstore import load_approved, load_emission, record_intent
from uadclaw.facts import ApkFacts
from uadclaw.factstore import store_device_facts
from uadclaw.ladder import Removal
from uadclaw.models import BranchEmission, JobKind, PackageAnalysis
from uadclaw.stages import StageInputError, branch_stage, pipeline_stage_handlers
from uadclaw.triagestore import record_decision
from uadclaw.worker import StageContext

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
MODEL = "deepseek-v4-flash"
BUNDLE = "b" * 64
LIST_PATH = "resources/assets/uad_lists.json"
PIPELINE_SHA = "9" * 40

# Two entries in the dominant key order plus one in the minority order, which is the shape the
# live file actually carries. A splice that reformatted would change the second entry's bytes.
BASE_LIST = {
    "com.already.upstream": {
        "list": "Oem",
        "description": "Already carried.",
        "dependencies": [],
        "neededBy": [],
        "labels": [],
        "removal": "Recommended",
    },
    "com.other.order": {
        "description": "A minority-order entry.",
        "removal": "Advanced",
        "list": "Oem",
        "dependencies": [],
        "neededBy": [],
        "labels": [],
    },
}


@pytest.fixture(autouse=True)
def _isolate_git_config(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args], capture_output=True, check=True, text=True, timeout=60
    )
    return result.stdout.strip()


def git_bytes(root: Path, ref: str, path: str) -> bytes:
    """A committed blob's exact bytes. `git show` through the text helper would strip the
    trailing newline, and the digest under test is over the bytes."""
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "cat-file", "blob", f"{ref}:{path}"],
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout


def branches(root: Path) -> list[str]:
    """Every local branch, sorted. `for-each-ref` orders by refname, so a name sorting after
    `main` and one sorting before it are not the same list."""
    listed = git(root, "for-each-ref", "--format=%(refname)", "refs/heads/")
    return sorted(line for line in listed.splitlines() if line)


@pytest.fixture
def clone(tmp_path) -> Path:
    """A clone shaped like the upstream repo: one commit on `main` carrying the list."""
    directory = tmp_path / "uad-clone"
    directory.mkdir(parents=True)
    subprocess.run(  # noqa: S603
        ["git", "init", "--quiet", "-b", "main", str(directory)],
        capture_output=True,
        check=True,
        timeout=60,
    )
    git(directory, "config", "user.name", "Clone Owner")
    git(directory, "config", "user.email", "owner@example.invalid")
    target = directory / LIST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(json.dumps(BASE_LIST, indent=2).encode())
    git(directory, "add", "--", LIST_PATH)
    git(directory, "commit", "--quiet", "-m", "seed")
    return directory


@pytest.fixture
def emission_env(monkeypatch, clone):
    monkeypatch.setenv("UPSTREAM_REPO_PATH", str(clone))
    monkeypatch.setenv("UPSTREAM_REPO_LIST_PATH", LIST_PATH)
    monkeypatch.setenv("UPSTREAM_BASE_REF", "main")
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", PIPELINE_SHA)
    return clone


# --- fixtures over the real writers ------------------------------------------------------------


def make_facts(package: str) -> ApkFacts:
    return ApkFacts(  # type: ignore[arg-type]
        package=package,
        label="Example",
        label_unresolved=False,
        version_code=1,
        partition="product",
        device_path=f"/product/app/{package}/{package}.apk",
        priv_app=False,
        sha256="0" * 64,
        cert_issuer="Organization: Example Corp",
        cert_subject="Organization: Example Corp",
        core_app=False,
        shared_user_id=None,
        persistent=False,
        has_code=True,
        overlay_target=None,
        overlay_static=False,
        overlay_priority=None,
    )


async def approve(
    session_factory,
    package: str,
    *,
    device_key: str = "pixel:oriole",
    floor: str = "Recommended",
    removal: Removal = Removal.ADVANCED,
) -> None:
    async with session_factory() as session, session.begin():
        await store_device_facts(
            session,
            device_key=device_key,
            build="bp1a.260505.001",
            facts=[make_facts(package)],
            observed_at=NOW,
        )
        session.add(
            PackageAnalysis(
                package=package,
                updated_at=NOW,
                queued=True,
                filter_verdict="queued",
                floor=floor,
                floor_rule="default",
                dependencies=[],
                needed_by=[],
            )
        )
    async with session_factory() as session, session.begin():
        await store_classification(
            session,
            Classification(
                package=package,
                bundle_sha256=BUNDLE,
                description=f"Vendor application {package}. Removing it loses its local data.",
                list=UadList.MISC,
                removal=removal,
                confidence=Confidence.MEDIUM,
                unknown_fields=(),
                reasoning_brief="No privileged surface.",
                provenance={"description": f"llm:{MODEL}", "removal": f"llm:{MODEL}"},
            ),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )
    async with session_factory() as session, session.begin():
        await record_decision(
            session, package=package, bundle_sha256=BUNDLE, action="approve", at=NOW
        )


@pytest.fixture
async def emission_db(db_session_factory):
    async with db_session_factory() as session, session.begin():
        await session.execute(text("TRUNCATE TABLE package_triage_decision RESTART IDENTITY"))
    return db_session_factory


async def make_job(session_factory, vendor: str = "pixel"):
    async with session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.BRANCH_EMISSION.value, params={"vendor": vendor}
        )
        return job.id


def context(job_id, session_factory) -> StageContext:
    # `scratch_dir=None` is what the worker hands a kind with JOB_KIND_NEEDS_SCRATCH False, so
    # a stage reaching for scratch would fail here rather than in production.
    return StageContext(job_id=job_id, attempt=1, scratch_dir=None, session_factory=session_factory)


# --- the happy path ----------------------------------------------------------------------------


async def test_a_vendor_batch_becomes_one_branch_carrying_one_appended_file(
    db_env, emission_db, emission_env
):
    """The deliverable: a branch off `main` with exactly one file changed, the existing bytes
    untouched, and the new entries appended in the dominant key order."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    await approve(emission_db, "com.example.two")
    job_id = await make_job(emission_db)
    base = git(clone, "rev-parse", "main")

    await branch_stage(context(job_id, emission_db))

    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    assert git(clone, "rev-parse", f"{branch}^") == base
    assert git(clone, "diff-tree", "--no-commit-id", "--name-only", "-r", branch) == LIST_PATH
    committed = git(clone, "show", f"{branch}:{LIST_PATH}")
    parsed = json.loads(committed)
    assert list(parsed) == [
        "com.already.upstream",
        "com.other.order",
        "com.example.one",
        "com.example.two",
    ]
    # Every pre-existing byte survives: reformatting `uad_lists.json` is a documented upstream
    # rejection reason, and "the JSON is equivalent" is exactly the assertion a reformat passes.
    original = json.dumps(BASE_LIST, indent=2)
    assert committed.startswith(original[: original.rindex("\n}")])
    assert list(parsed["com.example.one"]) == [
        "list",
        "description",
        "dependencies",
        "neededBy",
        "labels",
        "removal",
    ]


async def test_the_run_is_recorded_with_the_body_and_every_package_that_shipped(
    db_env, emission_db, emission_env
):
    """The row answers "what exactly went into that PR" a month later, once
    `package_classification` has been rewritten by the next model run."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)

    await branch_stage(context(job_id, emission_db))

    async with emission_db() as session:
        row = (await session.execute(select(BranchEmission))).scalar_one()
        record = await load_emission(session, job_id=job_id)
    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    assert record is not None
    assert row.commit_oid == git(clone, "rev-parse", branch)
    assert row.base_commit == git(clone, "rev-parse", "main")
    assert row.reconciled is False
    assert row.package_count == 1
    assert row.vendor == "pixel"
    assert row.pipeline_commit_sha == PIPELINE_SHA
    # The digest is over the exact bytes committed, read back as BYTES: it is the key the
    # crash recovery matches a found branch against, so a hash of anything but the blob would
    # make every recovery refuse a branch it had itself written.
    assert row.list_sha256 == hashlib.sha256(git_bytes(clone, branch, LIST_PATH)).hexdigest()
    # The disclosure upstream's CONTRIBUTING demands, on the row rather than on disk: the web
    # container has a read-only rootfs and a second copy is somewhere this can drift from.
    assert PIPELINE_SHA in row.pr_body
    assert "generated by a large language model" in row.pr_body
    assert MODEL in row.pr_body


async def test_the_commit_message_follows_upstreams_own_convention(
    db_env, emission_db, emission_env
):
    """It is read by that project's maintainers, whose merged package PRs are `pkg(...)`, so
    it is their convention that applies here and not this repo's."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)

    await branch_stage(context(job_id, emission_db))

    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    assert git(clone, "log", "-1", "--format=%s", branch) == "pkg(pixel): add 1 package(s)"


async def test_nothing_is_pushed_fetched_or_authenticated(db_env, emission_db, emission_env):
    """Design decision 8: there is no GitHub credential in this stack, so the pipeline's last
    act is a local commit. Asserted on the clone's own state — it has no remote at all, so a
    push or fetch anywhere in the path would have failed rather than succeeded quietly."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)

    await branch_stage(context(job_id, emission_db))

    assert git(clone, "remote") == ""
    assert git(clone, "for-each-ref", "--format=%(refname)", "refs/remotes/") == ""


async def test_the_emission_is_pinned_to_one_commit_not_to_a_moving_ref(
    db_env, emission_db, emission_env, monkeypatch
):
    """`emit_branch` re-resolves whatever ref it is handed, so passing `main` twice would let a
    fetch landing between the inspection and the commit rebase the emission onto a newer base
    while the spliced bytes still came from the older blob — which silently reverts whatever
    arrived in between and looks exactly like a correct branch.

    The window is forced by moving `main` between the two calls, which is what a fetch does.
    """
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    base = git(clone, "rev-parse", "main")

    import uadclaw.stages as stages_module

    original_insert = stages_module.insert_entries

    def insert_then_let_main_move(raw, packages):
        spliced = original_insert(raw, packages)
        moved = dict(BASE_LIST)
        moved["com.landed.meanwhile"] = {"list": "Oem", "removal": "Recommended"}
        (clone / LIST_PATH).write_bytes(json.dumps(moved, indent=2).encode())
        git(clone, "add", "--", LIST_PATH)
        git(clone, "commit", "--quiet", "-m", "somebody fetched")
        return spliced

    monkeypatch.setattr(stages_module, "insert_entries", insert_then_let_main_move)
    await branch_stage(context(job_id, emission_db))

    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    assert git(clone, "rev-parse", f"{branch}^") == base
    assert git(clone, "rev-parse", "main") != base


# --- what the stage refuses ----------------------------------------------------------------------


async def test_no_configured_clone_refuses_naming_the_setting(
    db_env, emission_db, monkeypatch, clone
):
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", PIPELINE_SHA)
    monkeypatch.delenv("UPSTREAM_REPO_PATH", raising=False)
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)

    with pytest.raises(StageInputError, match="UPSTREAM_REPO_PATH"):
        await branch_stage(context(job_id, emission_db))


async def test_an_unrecorded_pipeline_commit_refuses_before_the_clone_is_touched(
    db_env, emission_db, emission_env, monkeypatch
):
    """The disclosure is mandatory upstream, so a body that cannot name the commit that
    produced the entries is a refusal rather than a line quietly left out. Refused before any
    git call, so a box with a misconfigured deploy never half-writes a branch."""
    clone = emission_env
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", "  ")
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)

    with pytest.raises(StageInputError, match="PIPELINE_COMMIT_SHA"):
        await branch_stage(context(job_id, emission_db))

    assert branches(clone) == ["refs/heads/main"]
    async with emission_db() as session:
        assert (await session.execute(select(BranchEmission))).scalars().all() == []


async def test_a_vendor_with_nothing_approved_refuses_rather_than_cutting_an_empty_branch(
    db_env, emission_db, emission_env
):
    clone = emission_env
    await approve(emission_db, "com.example.one", device_key="samsung:SM-S911U")
    job_id = await make_job(emission_db, vendor="pixel")

    with pytest.raises(StageInputError, match="nothing is approved under vendor 'pixel'"):
        await branch_stage(context(job_id, emission_db))

    assert branches(clone) == ["refs/heads/main"]


async def test_a_bad_vendor_is_refused_at_job_creation_and_never_reaches_a_worker(emission_db):
    """422 at creation rather than a claimed job: the vendor becomes a PR heading and a branch
    segment, so a newline or a slash in it is decided before either exists."""
    async with emission_db() as session, session.begin():
        for bad in ("", "pixel\n## injected", "pixel/../etc", "-pixel"):
            with pytest.raises(jobs_module.JobValidationError):
                await jobs_module.create_job(
                    session, kind=JobKind.BRANCH_EMISSION.value, params={"vendor": bad}
                )


# --- the crash window --------------------------------------------------------------------------


async def test_a_finished_job_re_run_cuts_no_second_branch(db_env, emission_db, emission_env):
    """The plain idempotence leg: a reclaim after the job completed must not emit again."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    await branch_stage(context(job_id, emission_db))
    before = git(clone, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")

    await branch_stage(context(job_id, emission_db))

    assert git(clone, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/") == before
    async with emission_db() as session:
        assert len((await session.execute(select(BranchEmission))).scalars().all()) == 1


async def test_a_finished_job_re_run_after_the_branch_was_pushed_and_deleted_stays_done(
    db_env, emission_db, emission_env
):
    """The leg the plain idempotence test cannot reach, and the one that decides whether the
    finished check is load-bearing at all.

    While the branch is still lying in the clone, disabling that check lands harmlessly in the
    recovery path, which recognises its own branch and no-ops. Once a human has done the thing
    this pipeline exists to hand them — push the branch and delete it — there is no branch left
    to recognise, and the same job re-emits: a SECOND branch for a batch already delivered.
    That is exactly the outcome the whole write-first ordering exists to prevent, and it is
    reachable, because a worker that crashes after the commit is reclaimed on the stale
    heartbeat minutes or hours later.
    """
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    await branch_stage(context(job_id, emission_db))
    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    async with emission_db() as session:
        emitted = (await load_emission(session, job_id=job_id)).commit_oid
    # What the operator does next: push, then tidy up.
    git(clone, "checkout", "--quiet", "main")
    git(clone, "branch", "--delete", "--force", branch)

    await branch_stage(context(job_id, emission_db))

    assert branches(clone) == ["refs/heads/main"]
    async with emission_db() as session:
        rows = (await session.execute(select(BranchEmission))).scalars().all()
    assert [row.commit_oid for row in rows] == [emitted]


async def test_a_crash_between_the_commit_and_its_recording_is_recovered(
    db_env, emission_db, emission_env, monkeypatch
):
    """THE case the write-first ordering exists for. `emit_branch` succeeded and the write
    after it never landed, so the retry meets a row with no outcome and a branch nobody
    recorded. It must recover the commit rather than cut a second branch — and `emit_branch`
    refuses an existing branch, so without the recovery this job is wedged forever.

    The crash is real rather than simulated by hand-writing a row: the first call runs the
    whole stage and dies at the recording write, exactly where a SIGKILL would.
    """
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)

    import uadclaw.stages as stages_module

    async def die_after_the_commit(*args, **kwargs):
        raise RuntimeError("the worker was killed here")

    monkeypatch.setattr(stages_module, "record_commit", die_after_the_commit)
    with pytest.raises(RuntimeError, match="killed here"):
        await branch_stage(context(job_id, emission_db))

    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    committed = git(clone, "rev-parse", branch)
    async with emission_db() as session:
        assert (await load_emission(session, job_id=job_id)).commit_oid is None

    monkeypatch.undo()
    monkeypatch.setenv("UPSTREAM_REPO_PATH", str(clone))
    monkeypatch.setenv("UPSTREAM_REPO_LIST_PATH", LIST_PATH)
    monkeypatch.setenv("UPSTREAM_BASE_REF", "main")
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", PIPELINE_SHA)
    await branch_stage(context(job_id, emission_db))

    # No second branch, and the row now says how it learned the commit.
    assert branches(clone) == ["refs/heads/main", f"refs/heads/{branch}"]
    async with emission_db() as session:
        row = (await session.execute(select(BranchEmission))).scalar_one()
    assert row.commit_oid == committed
    assert row.reconciled is True


async def test_a_recovery_refuses_a_branch_that_is_not_the_one_this_job_wrote(
    db_env, emission_db, emission_env
):
    """The recovery keys on the digest the intent recorded, so a branch of the same name
    carrying something else is refused rather than adopted. Reachable: the branch name is
    derived from the job, and a human can create any ref they like in their own clone."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    packages = None
    async with emission_db() as session:
        packages = await load_approved(session, vendor="pixel")
    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    async with emission_db() as session, session.begin():
        await record_intent(
            session,
            job_id=job_id,
            vendor="pixel",
            branch=branch,
            repo_path=str(clone),
            list_path=LIST_PATH,
            base_commit=git(clone, "rev-parse", "main"),
            list_sha256="d" * 64,
            pipeline_version="0.1.0",
            pipeline_commit_sha=PIPELINE_SHA,
            pr_body="## pixel",
            packages=packages,
            at=NOW,
        )
    git(clone, "branch", branch, "main")

    with pytest.raises(StageInputError, match="not the .* this job's own emission recorded"):
        await branch_stage(context(job_id, emission_db))

    async with emission_db() as session:
        assert (await load_emission(session, job_id=job_id)).commit_oid is None


async def test_an_intent_whose_branch_was_rolled_back_is_re_emitted(
    db_env, emission_db, emission_env
):
    """`emit_branch` deletes the branch it created on every failure path, so a row with no
    outcome and no branch means the emission never landed. The retry re-derives the batch from
    what is approved NOW rather than replaying the recorded one, which is what lets a fix
    upstream of here (an approval added, a floor computed) actually take effect."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    async with emission_db() as session:
        packages = await load_approved(session, vendor="pixel")
    async with emission_db() as session, session.begin():
        await record_intent(
            session,
            job_id=job_id,
            vendor="pixel",
            branch=branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id)),
            repo_path=str(clone),
            list_path=LIST_PATH,
            base_commit=git(clone, "rev-parse", "main"),
            list_sha256="d" * 64,
            pipeline_version="0.1.0",
            pipeline_commit_sha=PIPELINE_SHA,
            pr_body="## pixel",
            packages=packages,
            at=NOW,
        )
    # The fix that lands between the failed attempt and the retry.
    await approve(emission_db, "com.example.two")

    await branch_stage(context(job_id, emission_db))

    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    parsed = json.loads(git(clone, "show", f"{branch}:{LIST_PATH}"))
    assert "com.example.two" in parsed
    async with emission_db() as session:
        row = (await session.execute(select(BranchEmission))).scalar_one()
    assert row.reconciled is False
    assert row.package_count == 2
    assert row.list_sha256 != "d" * 64


async def test_two_attempts_of_one_job_derive_the_same_branch_name():
    """What makes every case above possible: the name is a function of the job, so a retry
    meets its own branch instead of cutting a neighbour. A date would collide between two
    batches for one vendor on one day, and a digest of the batch would move exactly when a
    retry must not — the moment a reviewer approves one more package."""
    job_id = "3f2a1b9c-4d5e-6789-abcd-ef0123456789"

    first = branch_name(prefix="uadclaw", vendor="pixel", job_id=job_id)
    second = branch_name(prefix="uadclaw", vendor="pixel", job_id=job_id.upper())

    assert first == second == "uadclaw/pixel-3f2a1b9c4d5e"
    assert branch_name(prefix="uadclaw", vendor="pixel", job_id="0" * 32) != first


@pytest.mark.parametrize(
    "vendor", ["pixel/../main", "pixel branch", "pixel\n## injected", "-pixel", ""]
)
def test_branch_name_refuses_a_vendor_that_would_name_a_different_branch(vendor):
    """A SECOND gate over the one `BranchEmissionJobParams` already applies, and it needs its
    own test for exactly that reason: with the params model in front of it, nothing production
    does can reach this check, so it was silent to the whole suite until asked directly.
    `branch_name` is public and pure, and the two gates guard different callers — a params
    model cannot bound a caller that never went through job creation."""
    with pytest.raises(EmissionError, match="not a driver name"):
        branch_name(prefix="uadclaw", vendor=vendor, job_id="0123456789abcdef")


@pytest.mark.parametrize("prefix", ["", "-uadclaw", "uad claw", "/uadclaw"])
def test_branch_name_refuses_a_prefix_git_would_misread(prefix):
    with pytest.raises(EmissionError, match="not a branch-name prefix"):
        branch_name(prefix=prefix, vendor="pixel", job_id="0123456789abcdef")


def test_a_job_id_too_short_to_identify_a_job_is_refused():
    """The name has to be derivable from the job so a retry cannot cut a second branch. A
    truncated id would make two jobs share a name, where the second silently fails on
    `emit_branch`'s already-exists refusal instead of emitting."""
    with pytest.raises(EmissionError, match="hex digits"):
        branch_name(prefix="uadclaw", vendor="pixel", job_id="abc")


# --- the walk ------------------------------------------------------------------------------------


def test_branch_has_a_handler_and_still_is_not_in_the_firmware_walk():
    """`branch` sat in `PIPELINE_STAGES` with no handler, so a firmware job no-opped through
    it. Registering a real one makes that arrangement dangerous rather than merely untidy: a
    global walk would now commit into somebody's repository the moment any firmware job
    finished. `JOB_KIND_STAGES` is what prevents it, and this is the assertion that says so."""
    assert "branch" in pipeline_stage_handlers()
    assert jobs_module.stages_for("firmware_analysis")[-1] == "rule_ladder"
    assert "branch" not in jobs_module.stages_for("firmware_analysis")
    assert "branch" not in jobs_module.stages_for("classification")
    assert jobs_module.next_stage("rule_ladder", "firmware_analysis") is None
