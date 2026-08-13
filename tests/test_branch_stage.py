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

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, text

from uadclaw import jobs as jobs_module
from uadclaw.classify import Classification, Confidence, UadList
from uadclaw.classifystore import store_classification
from uadclaw.emission import EmissionError, branch_name
from uadclaw.emissionstore import load_emission
from uadclaw.facts import ApkFacts
from uadclaw.factstore import store_device_facts
from uadclaw.ladder import Removal
from uadclaw.models import BranchEmission, JobKind, PackageAnalysis
from uadclaw.settings import get_settings
from uadclaw.stages import StageInputError, branch_stage, pipeline_stage_handlers
from uadclaw.triagestore import record_decision
from uadclaw.upstreamrepo import UpstreamRepoError
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


def install_hook(root: Path, body: str, *, name: str = "pre-commit") -> None:
    hook = root / ".git" / "hooks" / name
    hook.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    hook.chmod(0o755)


async def crash_after_the_commit(session_factory, clone: Path, job_id, monkeypatch) -> str:
    """Run the stage for real and kill it exactly where the recording write happens.

    Leaves what a SIGKILL in that window leaves: the branch committed, the clone checked out
    ON it (which `emit_branch` promises, and which is why a human's next commit lands there),
    and the emission row still pending.

    The patch is undone by restoring the ONE attribute rather than with `monkeypatch.undo()`,
    which would also revert the autouse `GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` isolation and
    let this box's global `commit-msg` hook fire inside the throwaway clone on any commit a
    test makes afterwards.
    """
    import uadclaw.stages as stages_module

    real_record_commit = stages_module.record_commit

    async def die_after_the_commit(*args, **kwargs):
        raise RuntimeError("the worker was killed here")

    monkeypatch.setattr(stages_module, "record_commit", die_after_the_commit)
    with pytest.raises(RuntimeError, match="killed here"):
        await branch_stage(context(job_id, session_factory))
    monkeypatch.setattr(stages_module, "record_commit", real_record_commit)
    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    assert git(clone, "symbolic-ref", "--short", "HEAD") == branch
    return branch


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


# --- concurrent emissions -----------------------------------------------------------------------


async def test_two_concurrent_emissions_for_different_vendors_both_succeed(
    db_env, emission_db, emission_env
):
    """Both jobs write into the ONE clone, so without a lock they interleave their
    checkout-and-commit and each fails the other's read-back with a message blaming "a commit
    hook, or another process". Serialized, the second waits and both branches land.
    """
    clone = emission_env
    await approve(emission_db, "com.example.pixel")
    await approve(emission_db, "com.example.samsung", device_key="samsung:SM-S911U")
    pixel_job = await make_job(emission_db, vendor="pixel")
    samsung_job = await make_job(emission_db, vendor="samsung")

    await asyncio.gather(
        branch_stage(context(pixel_job, emission_db)),
        branch_stage(context(samsung_job, emission_db)),
    )

    pixel_branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(pixel_job))
    samsung_branch = branch_name(prefix="uadclaw", vendor="samsung", job_id=str(samsung_job))
    assert branches(clone) == sorted(
        ["refs/heads/main", f"refs/heads/{pixel_branch}", f"refs/heads/{samsung_branch}"]
    )


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

    branch = await crash_after_the_commit(emission_db, emission_env, job_id, monkeypatch)
    committed = git(clone, "rev-parse", branch)
    async with emission_db() as session:
        assert (await load_emission(session, job_id=job_id)).commit_oid is None

    await branch_stage(context(job_id, emission_db))

    # No second branch, and the row now says how it learned the commit.
    assert branches(clone) == ["refs/heads/main", f"refs/heads/{branch}"]
    async with emission_db() as session:
        row = (await session.execute(select(BranchEmission))).scalar_one()
    assert row.commit_oid == committed
    assert row.reconciled is True


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


# --- the recovery bounds the BRANCH, not one file's bytes ----------------------------------------


async def test_a_human_commit_riding_along_in_the_crash_window_is_never_adopted(
    db_env, emission_db, emission_env, monkeypatch
):
    """The blocker this round fixed, and the exact shape `emit_branch` was already hardened
    against once: a read-back that bounds the COMMIT does not bound the BRANCH.

    A successful emission leaves the clone checked out ON the emission branch — that is
    `emit_branch`'s documented contract, because the human's next act is `git push` from there.
    So in the crash window a human working in their own clone commits, and their commit lands
    on OUR branch. It touches nothing this pipeline wrote, so the list blob still hashes to the
    recorded digest and a content-only recovery adopts their commit as this batch's outcome.
    The operator is then told to push a branch carrying an unrelated file, disclosed upstream
    as this pipeline's work.

    Bounding the parent is what catches it, so this asserts the branch is refused AND that
    nothing was recorded.
    """
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    branch = await crash_after_the_commit(emission_db, clone, job_id, monkeypatch)
    ours = git(clone, "rev-parse", branch)
    # The human, working in their own clone, on the branch the emission left checked out.
    (clone / "NOTES.md").write_text("my own notes\n", encoding="utf-8")
    git(clone, "add", "--", "NOTES.md")
    git(clone, "commit", "--quiet", "-m", "my own work")
    theirs = git(clone, "rev-parse", branch)
    assert theirs != ours

    with pytest.raises(UpstreamRepoError, match="not the single commit"):
        await branch_stage(context(job_id, emission_db))

    async with emission_db() as session:
        row = (await session.execute(select(BranchEmission))).scalar_one()
    assert row.commit_oid is None
    assert row.reconciled is False


async def test_an_amended_commit_that_adds_a_file_is_never_adopted(
    db_env, emission_db, emission_env, monkeypatch
):
    """The shape the parent check alone cannot catch, which is why the touched-paths read-back
    is a separate leg: amending keeps the parent AND the list bytes, and only the file list
    changes."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    branch = await crash_after_the_commit(emission_db, clone, job_id, monkeypatch)
    (clone / "NOTES.md").write_text("my own notes\n", encoding="utf-8")
    git(clone, "add", "--", "NOTES.md")
    git(clone, "commit", "--quiet", "--amend", "--no-edit")
    assert git(clone, "rev-parse", f"{branch}^") == git(clone, "rev-parse", "main")

    with pytest.raises(UpstreamRepoError, match="touches"):
        await branch_stage(context(job_id, emission_db))

    async with emission_db() as session:
        assert (await load_emission(session, job_id=job_id)).commit_oid is None


async def test_a_branch_carrying_different_bytes_is_never_adopted(
    db_env, emission_db, emission_env, monkeypatch
):
    """The third leg. A hook that reformats `uad_lists.json` after the commit is a documented
    upstream rejection reason, so a branch whose blob moved is not this emission's outcome."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    await crash_after_the_commit(emission_db, clone, job_id, monkeypatch)
    rewritten = json.loads((clone / LIST_PATH).read_text())
    (clone / LIST_PATH).write_text(json.dumps(rewritten, indent=4))
    git(clone, "add", "--", LIST_PATH)
    git(clone, "commit", "--quiet", "--amend", "--no-edit")

    with pytest.raises(UpstreamRepoError, match="sha256"):
        await branch_stage(context(job_id, emission_db))

    async with emission_db() as session:
        assert (await load_emission(session, job_id=job_id)).commit_oid is None


async def test_a_recovery_against_a_repointed_clone_says_so(
    db_env, emission_db, emission_env, monkeypatch, tmp_path
):
    """`EmissionRecord` carries the work-tree root the emission actually happened in, because
    `UPSTREAM_REPO_PATH` is an operator setting that can be repointed between a crash and the
    retry. The verification would nearly always fail first on a different clone; this is the
    leg that names what changed instead."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    await crash_after_the_commit(emission_db, clone, job_id, monkeypatch)
    # A byte-identical second checkout, so every branch read-back passes there too and the
    # path check is the only thing left that can notice.
    twin = tmp_path / "twin"
    shutil.copytree(clone, twin)
    monkeypatch.setenv("UPSTREAM_REPO_PATH", str(twin))
    # `get_settings` is an `lru_cache`, and the first `branch_stage` call above already
    # populated it. Without this the setenv repoints nothing, the stage keeps reading the
    # original clone, and the test passes for the wrong reason — which is exactly how a probe
    # of this concern came back inconclusive during review.
    get_settings.cache_clear()

    with pytest.raises(StageInputError, match="has been repointed"):
        await branch_stage(context(job_id, emission_db))

    async with emission_db() as session:
        assert (await load_emission(session, job_id=job_id)).commit_oid is None


# --- the ambiguous state refuses rather than double-shipping -------------------------------------


async def test_a_pending_intent_with_no_branch_refuses_instead_of_emitting_again(
    db_env, emission_db, emission_env, monkeypatch
):
    """Pending row, no branch, and two histories produce it: an emission that died before
    committing anything, or one that committed, got pushed, and had its branch deleted before
    the recording landed. Nothing inside the clone separates them and they want opposite
    actions, so the ambiguity resolves toward the refusal — guessing the other way sends a
    batch upstream twice.

    The push-and-delete history is the one driven here, because it is the expensive one.
    """
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    branch = await crash_after_the_commit(emission_db, clone, job_id, monkeypatch)
    # What the operator does with a branch they were handed.
    git(clone, "checkout", "--quiet", "main")
    git(clone, "branch", "--delete", "--force", branch)

    import uadclaw.stages as stages_module

    calls: list[str] = []
    monkeypatch.setattr(
        stages_module,
        "emit_branch",
        lambda *args, **kwargs: calls.append(kwargs.get("branch", "?")),
    )
    with pytest.raises(StageInputError, match="outcome was never written"):
        await branch_stage(context(job_id, emission_db))

    assert calls == []
    assert branches(clone) == ["refs/heads/main"]
    async with emission_db() as session:
        rows = (await session.execute(select(BranchEmission))).scalars().all()
    assert len(rows) == 1
    assert rows[0].commit_oid is None


async def test_an_ordinary_failed_emission_stays_retryable(db_env, emission_db, emission_env):
    """The other half, and the reason the ambiguous case above can be refused at all: a
    `pre-commit` hook rejecting the commit is an everyday operator-fixable failure, and the
    retry after they fix it has to work.

    `emit_branch` rolls its branch back on every failure path, and that rollback is READ BACK
    rather than trusted: the intent is discarded only once the branch is confirmed gone, so a
    retryable failure leaves no row and the retry takes the ordinary path.
    """
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)
    install_hook(clone, "exit 1")

    with pytest.raises(UpstreamRepoError):
        await branch_stage(context(job_id, emission_db))

    async with emission_db() as session:
        assert await load_emission(session, job_id=job_id) is None
    assert branches(clone) == ["refs/heads/main"]

    # The operator fixes the clone and re-runs.
    install_hook(clone, "exit 0")
    await branch_stage(context(job_id, emission_db))

    branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(job_id))
    assert branches(clone) == ["refs/heads/main", f"refs/heads/{branch}"]
    async with emission_db() as session:
        row = (await session.execute(select(BranchEmission))).scalar_one()
    assert row.commit_oid == git(clone, "rev-parse", branch)
    assert row.reconciled is False


async def test_a_failed_emission_whose_branch_survived_keeps_its_intent(
    db_env, emission_db, emission_env, monkeypatch
):
    """The case that makes the read-back necessary rather than decorative. `emit_branch`
    promises a rollback and there is one shape where it cannot deliver — a rollback that itself
    failed — and its own error text is prose nobody should be matching on. A branch still
    standing means the intent must survive so the next run can verify and adopt it."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    job_id = await make_job(emission_db)

    import uadclaw.stages as stages_module

    def emit_and_leave_the_branch(path, **kwargs):
        git(clone, "branch", kwargs["branch"], "main")
        raise UpstreamRepoError("emit_branch: and the clone could not be put back")

    monkeypatch.setattr(stages_module, "emit_branch", emit_and_leave_the_branch)
    with pytest.raises(UpstreamRepoError):
        await branch_stage(context(job_id, emission_db))

    async with emission_db() as session:
        record = await load_emission(session, job_id=job_id)
    assert record is not None
    assert record.commit_oid is None


# --- a vendor can be emitted more than once ------------------------------------------------------


async def test_a_second_batch_ships_after_the_first_one_merged_upstream(
    db_env, emission_db, emission_env
):
    """Nothing retires an approved package once it has shipped: the classification row and the
    human's `approve` both survive emission. So once the first batch merges and the operator
    pulls, a naive second emission hands `insert_entries` the packages it already sent and is
    refused WHOLE — new approvals included — making emission single-shot per vendor forever.

    The already-carried set is decided against the destination's own bytes at the base commit,
    which is the only copy that can answer it.
    """
    clone = emission_env
    await approve(emission_db, "com.example.one")
    first_job = await make_job(emission_db)
    await branch_stage(context(first_job, emission_db))
    first_branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(first_job))
    # Upstream merges it and the operator pulls.
    merged = git_bytes(clone, first_branch, LIST_PATH)
    git(clone, "checkout", "--quiet", "main")
    (clone / LIST_PATH).write_bytes(merged)
    git(clone, "add", "--", LIST_PATH)
    git(clone, "commit", "--quiet", "-m", "merged upstream")
    await approve(emission_db, "com.example.two")
    second_job = await make_job(emission_db)

    await branch_stage(context(second_job, emission_db))

    second_branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(second_job))
    parsed = json.loads(git_bytes(clone, second_branch, LIST_PATH))
    assert "com.example.two" in parsed
    # Exactly once, not twice: the already-carried one was dropped rather than re-proposed.
    assert list(parsed).count("com.example.one") == 1
    async with emission_db() as session:
        rows = (await session.execute(select(BranchEmission).order_by(BranchEmission.id))).scalars()
        counts = [row.package_count for row in rows]
    assert counts == [1, 1]


async def test_a_vendor_whose_whole_batch_already_shipped_refuses_clearly(
    db_env, emission_db, emission_env
):
    """The end state of the loop above. It is a refusal rather than a silent success because a
    human queued a job asking for a branch and there is none to cut, but the message has to say
    which of the two "nothing to emit" reasons it is."""
    clone = emission_env
    await approve(emission_db, "com.example.one")
    first_job = await make_job(emission_db)
    await branch_stage(context(first_job, emission_db))
    first_branch = branch_name(prefix="uadclaw", vendor="pixel", job_id=str(first_job))
    merged = git_bytes(clone, first_branch, LIST_PATH)
    git(clone, "checkout", "--quiet", "main")
    (clone / LIST_PATH).write_bytes(merged)
    git(clone, "add", "--", LIST_PATH)
    git(clone, "commit", "--quiet", "-m", "merged upstream")
    second_job = await make_job(emission_db)

    with pytest.raises(StageInputError, match="already carried"):
        await branch_stage(context(second_job, emission_db))

    async with emission_db() as session:
        rows = (await session.execute(select(BranchEmission))).scalars().all()
    assert len(rows) == 1
