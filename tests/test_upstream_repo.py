"""Branch emission against real throwaway clones, never a mock of `subprocess`.

A mock would pin this module's idea of what git prints, which is the half most likely to be
wrong: `git checkout refs/heads/x` detaches, `check-ref-format --branch '@{-1}'` expands and
exits 0, a plain checkout carries a staged edit onto the branch it switches to. Every repo here
is built by `git init` under `tmp_path` and read back with git.

The clones are isolated from this box's own git configuration (`GIT_CONFIG_GLOBAL`,
`GIT_CONFIG_SYSTEM`) rather than merely given a local identity: a global `core.hooksPath` — which
this box does set — otherwise runs a commit-message linter inside every test repo, so the suite
would pass or fail on whichever machine ran it.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from uadclaw.upstreamrepo import (
    UpstreamRepoError,
    branch_exists,
    emit_branch,
    inspect_repo,
)

LIST_PATH = "resources/assets/uad_lists.json"
BASE_LIST = {
    "com.example.one": {"list": "Unsafe", "removal": "Expert", "description": "one"},
    "com.example.two": {"list": "Aosp", "removal": "Recommended", "description": "two"},
}
NEW_LIST = {**BASE_LIST, "com.example.three": {"list": "Oem", "removal": "Advanced"}}


def _bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=4).encode("utf-8")


@pytest.fixture(autouse=True)
def _isolate_git_config(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def git_code(root: Path, *args: str) -> int:
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
        timeout=60,
    ).returncode


def make_clone(directory: Path, *, payload: object = BASE_LIST, list_path: str = LIST_PATH) -> Path:
    """A clone shaped like the upstream repo: one commit on `base` carrying the list file."""
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        ["git", "init", "--quiet", "-b", "base", str(directory)],
        capture_output=True,
        check=True,
        timeout=60,
    )
    git(directory, "config", "user.name", "Clone Owner")
    git(directory, "config", "user.email", "owner@example.invalid")
    target = directory / list_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_bytes(payload))
    git(directory, "add", "--", list_path)
    git(directory, "commit", "--quiet", "-m", "seed")
    return directory


@pytest.fixture
def clone(tmp_path) -> Path:
    return make_clone(tmp_path / "uad-clone")


def install_hook(root: Path, body: str, *, name: str = "pre-commit") -> None:
    hook = root / ".git" / "hooks" / name
    hook.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    hook.chmod(0o755)


# --- inspect_repo: what makes a clone usable -----------------------------------------------


def test_a_clone_is_identified_by_the_list_it_carries_and_never_by_its_remote(clone):
    """The clone may be a fork, so its remote URL, its remote name and its branch names are all
    things somebody else renamed. The list file at the configured path is the identity."""
    git(clone, "remote", "add", "upstream", "https://example.invalid/someone-else/their-fork.git")
    git(clone, "branch", "--move", "base", "their-default")

    state = inspect_repo(clone, base_ref="their-default", list_path=LIST_PATH)

    assert state.list_bytes == _bytes(BASE_LIST)
    assert state.base_commit == git(clone, "rev-parse", "their-default")
    assert state.list_path == LIST_PATH
    assert state.path == str(clone)


def test_a_clone_with_no_remote_at_all_is_still_accepted(clone):
    assert git(clone, "remote") == ""

    assert inspect_repo(clone, base_ref="base", list_path=LIST_PATH).list_bytes == _bytes(BASE_LIST)


def test_the_list_is_read_at_the_base_ref_and_not_off_the_working_tree(clone):
    """What an emission edits has to be what the branch is based on: a clone parked on some
    other branch must not contribute its own copy of the file."""
    first = git(clone, "rev-parse", "HEAD")
    (clone / LIST_PATH).write_bytes(_bytes({"com.later.commit": {}}))
    git(clone, "add", "--", LIST_PATH)
    git(clone, "commit", "--quiet", "-m", "later")

    state = inspect_repo(clone, base_ref=first, list_path=LIST_PATH)

    assert state.list_bytes == _bytes(BASE_LIST)
    assert (clone / LIST_PATH).read_bytes() == _bytes({"com.later.commit": {}})


def test_a_path_that_is_not_a_directory_is_refused(tmp_path):
    with pytest.raises(UpstreamRepoError, match="is not a directory"):
        inspect_repo(tmp_path / "absent", base_ref="base", list_path=LIST_PATH)


def test_a_directory_that_is_not_a_git_work_tree_is_refused(tmp_path):
    plain = tmp_path / "just-files"
    (plain / "resources" / "assets").mkdir(parents=True)
    (plain / LIST_PATH).write_bytes(_bytes(BASE_LIST))

    with pytest.raises(UpstreamRepoError, match="is not a git work tree"):
        inspect_repo(plain, base_ref="base", list_path=LIST_PATH)


def test_a_dirty_work_tree_is_refused_and_the_error_names_the_file(clone):
    """Committing on top of somebody's half-finished work would bundle it into the PR."""
    (clone / LIST_PATH).write_bytes(_bytes({"com.someones.edit": {}}))

    with pytest.raises(UpstreamRepoError, match="uncommitted changes") as raised:
        inspect_repo(clone, base_ref="base", list_path=LIST_PATH)
    assert LIST_PATH in str(raised.value)


def test_an_untracked_file_counts_as_a_dirty_work_tree(clone):
    (clone / "notes.txt").write_text("half a thought", encoding="utf-8")

    with pytest.raises(UpstreamRepoError, match="uncommitted changes"):
        inspect_repo(clone, base_ref="base", list_path=LIST_PATH)


def test_a_base_ref_the_clone_does_not_carry_is_refused_by_name(clone):
    with pytest.raises(UpstreamRepoError, match="does not resolve to a commit") as raised:
        inspect_repo(clone, base_ref="main", list_path=LIST_PATH)
    assert "'main'" in str(raised.value)


def test_a_base_ref_starting_with_a_dash_is_refused_before_git_sees_it(clone):
    with pytest.raises(UpstreamRepoError, match="starts with a dash"):
        inspect_repo(clone, base_ref="--upload-pack=touch /tmp/pwned", list_path=LIST_PATH)


def test_a_list_path_absent_at_the_base_ref_is_refused(clone):
    with pytest.raises(UpstreamRepoError, match="does not exist at") as raised:
        inspect_repo(clone, base_ref="base", list_path="resources/assets/other.json")
    assert "resources/assets/other.json" in str(raised.value)


def test_a_list_path_naming_a_directory_is_refused_as_not_a_file(clone):
    """A parse-time check would raise on the same input one step later, calling a directory
    listing malformed JSON and sending the operator after the wrong fix."""
    with pytest.raises(UpstreamRepoError, match="is a tree, not a file") as raised:
        inspect_repo(clone, base_ref="base", list_path="resources/assets")
    assert "not valid JSON" not in str(raised.value)


def test_a_list_that_does_not_parse_is_refused(tmp_path):
    clone = make_clone(tmp_path / "broken")
    (clone / LIST_PATH).write_bytes(b'{"com.truncated": ')
    git(clone, "add", "--", LIST_PATH)
    git(clone, "commit", "--quiet", "-m", "truncate")

    with pytest.raises(UpstreamRepoError, match="not valid JSON"):
        inspect_repo(clone, base_ref="base", list_path=LIST_PATH)


def test_a_list_that_is_not_a_json_object_is_refused(tmp_path):
    clone = make_clone(tmp_path / "array", payload=["com.example.one"])

    with pytest.raises(UpstreamRepoError, match="parsed to list, not an object"):
        inspect_repo(clone, base_ref="base", list_path=LIST_PATH)


def test_an_empty_list_is_refused(tmp_path):
    clone = make_clone(tmp_path / "empty", payload={})

    with pytest.raises(UpstreamRepoError, match="carries zero entries"):
        inspect_repo(clone, base_ref="base", list_path=LIST_PATH)


def test_a_list_path_climbing_out_of_the_clone_is_refused(clone):
    with pytest.raises(UpstreamRepoError, match="climbs out of the clone"):
        inspect_repo(clone, base_ref="base", list_path="../../etc/passwd")


def test_a_list_path_that_is_a_symlink_out_of_the_clone_is_refused(tmp_path):
    """`resolve()` follows the link, so containment is asserted against where the write would
    actually land rather than against the string that named it."""
    outside = tmp_path / "outside.json"
    outside.write_bytes(_bytes(BASE_LIST))
    clone = make_clone(tmp_path / "linked")
    (clone / LIST_PATH).unlink()
    (clone / LIST_PATH).symlink_to(outside)
    git(clone, "add", "--", LIST_PATH)
    git(clone, "commit", "--quiet", "-m", "link it")

    with pytest.raises(UpstreamRepoError, match="outside the clone"):
        inspect_repo(clone, base_ref="base", list_path=LIST_PATH)


def test_a_subdirectory_of_the_clone_resolves_to_the_work_tree_root(clone):
    """`git cat-file <rev>:<path>` always resolves from the work tree root, so the path that
    gets written has to be rooted there too or the two describe different files."""
    state = inspect_repo(clone / "resources", base_ref="base", list_path=LIST_PATH)

    assert state.path == str(clone)
    assert state.list_bytes == _bytes(BASE_LIST)


def test_a_stray_git_dir_in_the_environment_cannot_retarget_the_clone(clone, tmp_path, monkeypatch):
    """`GIT_DIR` overrides `git -C <path>` rather than combining with it. The global pre-commit
    hook exports its sibling `GIT_INDEX_FILE`, so this arrives from a real mechanism.

    Both shas are read BEFORE the variable is set: this test's own helper is a plain `git -C`
    too, and it gets retargeted just as hard — measured here, it answered for the other repo.
    """
    other = make_clone(tmp_path / "somebody-elses", payload={"com.other.repo": {}})
    expected = git(clone, "rev-parse", "base")
    decoy = git(other, "rev-parse", "base")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))

    state = inspect_repo(clone, base_ref="base", list_path=LIST_PATH)

    assert state.base_commit == expected
    assert state.base_commit != decoy
    assert state.list_bytes == _bytes(BASE_LIST)


def _index_identity(index_file: Path) -> tuple[int, int, int]:
    stat = index_file.stat()
    return (stat.st_mtime_ns, stat.st_ino, stat.st_size)


def test_inspecting_a_clone_does_not_rewrite_its_index(tmp_path):
    """The read-only claim, pinned on the thing an operator would actually notice: a plain
    `git status` refreshes a stale stat-cache and REWRITES `.git/index` doing it.

    The fixture has to be stale in mtime ONLY — same bytes, backdated stamp — and wide enough
    that the refresh has something to do; one file proves nothing. Measured on git 2.55: plain
    `status --porcelain` rewrites the index here, `--no-optional-locks status --porcelain` does
    not. A read-only `.git` does NOT separate the two (both exit 0 and git degrades quietly), so
    the write is the observable and the lock is not.
    """
    clone = make_clone(tmp_path / "wide")
    bulk = clone / "many"
    bulk.mkdir()
    for number in range(200):
        (bulk / f"f{number}.txt").write_text(f"{number}\n", encoding="utf-8")
    git(clone, "add", "--", "many")
    git(clone, "commit", "--quiet", "-m", "many files")
    stale = 1_600_000_000
    for path in sorted(bulk.iterdir()):
        os.utime(path, (stale, stale))
    index_file = clone / ".git" / "index"
    before = _index_identity(index_file)

    inspect_repo(clone, base_ref="base", list_path=LIST_PATH)

    assert _index_identity(index_file) == before


def test_inspecting_a_clone_never_moves_head_or_touches_the_tree(clone):
    """Safe against a clone a human is sitting in."""
    before = (git(clone, "rev-parse", "HEAD"), git(clone, "symbolic-ref", "HEAD"))

    inspect_repo(clone, base_ref="base", list_path=LIST_PATH)
    branch_exists(clone, "uadclaw/pixel-2026-08-13")

    assert (git(clone, "rev-parse", "HEAD"), git(clone, "symbolic-ref", "HEAD")) == before
    assert git(clone, "status", "--porcelain") == ""


# --- branch_exists ---------------------------------------------------------------------------


def test_branch_exists_answers_for_a_present_and_an_absent_branch(clone):
    git(clone, "branch", "uadclaw/already-here", "base")

    assert branch_exists(clone, "uadclaw/already-here") is True
    assert branch_exists(clone, "uadclaw/not-yet") is False


def test_a_branch_name_git_would_read_as_an_option_is_refused(clone):
    with pytest.raises(UpstreamRepoError, match="starts with a dash"):
        branch_exists(clone, "--force")


def test_an_invalid_branch_name_is_refused(clone):
    with pytest.raises(UpstreamRepoError, match="check-ref-format"):
        branch_exists(clone, "uadclaw/two words")


def test_a_shorthand_that_git_expands_is_refused_rather_than_taken_literally(clone):
    """`git check-ref-format --branch '@{-1}'` prints the previously checked-out branch and
    exits 0, so a validator reading only the exit code accepts a name that is not a name."""
    git(clone, "checkout", "--quiet", "-b", "somewhere-else")
    git(clone, "checkout", "--quiet", "base")

    with pytest.raises(UpstreamRepoError, match="expands to 'somewhere-else'"):
        branch_exists(clone, "@{-1}")


# --- emit_branch: the happy path ---------------------------------------------------------------


def test_it_commits_the_new_bytes_on_a_branch_cut_from_the_base_ref(clone):
    base_commit = git(clone, "rev-parse", "base")

    head = emit_branch(
        clone,
        branch="uadclaw/pixel-2026-08-13",
        base_ref="base",
        list_path=LIST_PATH,
        new_bytes=_bytes(NEW_LIST),
        message="feat(lists): add one pixel package",
    )

    assert head == git(clone, "rev-parse", "uadclaw/pixel-2026-08-13")
    assert git(clone, "rev-parse", "uadclaw/pixel-2026-08-13^") == base_commit
    assert git(clone, "log", "-1", "--format=%s", head) == "feat(lists): add one pixel package"
    assert git(clone, "show", f"{head}:{LIST_PATH}").encode("utf-8") == _bytes(NEW_LIST).strip()


def test_the_committed_bytes_are_exactly_the_bytes_handed_in(clone):
    """Byte-for-byte, because upstream reads a reformat of this file as a rejection reason: no
    added newline, no re-indent, no key reordering."""
    payload = b'{"com.example.one": {"removal": "Expert"},"com.example.two":{}}'

    head = emit_branch(
        clone,
        branch="uadclaw/exact",
        base_ref="base",
        list_path=LIST_PATH,
        new_bytes=payload,
        message="feat(lists): keep the bytes",
    )

    assert git(clone, "cat-file", "blob", f"{head}:{LIST_PATH}").encode("utf-8") == payload.strip()
    assert (clone / LIST_PATH).read_bytes() == payload


def test_it_leaves_the_clone_on_the_new_branch_with_a_clean_tree(clone):
    """The human's next act is `git push` from there."""
    emit_branch(
        clone,
        branch="uadclaw/ready-to-push",
        base_ref="base",
        list_path=LIST_PATH,
        new_bytes=_bytes(NEW_LIST),
        message="feat(lists): a batch",
    )

    assert git(clone, "symbolic-ref", "--short", "HEAD") == "uadclaw/ready-to-push"
    assert git(clone, "status", "--porcelain") == ""


def test_the_base_branch_is_left_carrying_its_own_content(clone):
    """Both halves, because the "nothing moved" half is true of an `emit_branch` that did
    nothing at all: an emission that returns a constant as its first statement passes every
    assertion below the positive two. That stub is also the exact shape of the `@` bug this test
    is named for, which moved `base` precisely because nothing else was watching it."""
    base_commit = git(clone, "rev-parse", "base")

    head = emit_branch(
        clone,
        branch="uadclaw/untouched-base",
        base_ref="base",
        list_path=LIST_PATH,
        new_bytes=_bytes(NEW_LIST),
        message="feat(lists): a batch",
    )

    assert branch_exists(clone, "uadclaw/untouched-base") is True
    assert head != base_commit
    assert git(clone, "rev-parse", "base") == base_commit
    assert git(clone, "show", f"base:{LIST_PATH}").encode("utf-8") == _bytes(BASE_LIST).strip()


def test_a_file_that_appears_mid_emission_is_not_swept_into_the_commit(clone):
    """`git add -- <path>`, never `-A` or `.`: the clone is a directory a human also works in,
    and their in-flight edit landing in an upstream PR is not recoverable by rebasing.

    A `post-checkout` hook is the deterministic form of that concurrent writer — it fires in
    exactly the window that matters, after the clean check and before the add.
    """
    install_hook(clone, "printf 'theirs' > theirs.txt", name="post-checkout")

    head = emit_branch(
        clone,
        branch="uadclaw/one-file-only",
        base_ref="base",
        list_path=LIST_PATH,
        new_bytes=_bytes(NEW_LIST),
        message="feat(lists): a batch",
    )

    assert git(clone, "diff-tree", "--no-commit-id", "--name-only", "-r", head) == LIST_PATH
    assert (clone / "theirs.txt").read_text() == "theirs"
    assert git(clone, "status", "--porcelain") == "?? theirs.txt"


def test_a_hook_writing_after_the_commit_leaves_the_branch_right_and_the_tree_dirty(clone):
    """A `post-commit` formatter writes into the working tree after the commit is already made.
    The COMMIT is what gets pushed and it is correct, so the emission reports success rather
    than discarding a good branch over a worktree-only edit — and the next emission then refuses
    the clone for dirt this one left, which is why the module docstring says so out loud instead
    of promising a clean tree unconditionally.
    """
    install_hook(clone, f"printf 'REFORMATTED BY HOOK' > {LIST_PATH}", name="post-commit")

    head = emit_branch(
        clone,
        branch="uadclaw/post-commit-hook",
        base_ref="base",
        list_path=LIST_PATH,
        new_bytes=_bytes(NEW_LIST),
        message="feat(lists): a batch",
    )

    assert (
        git(clone, "cat-file", "blob", f"{head}:{LIST_PATH}").encode("utf-8")
        == _bytes(NEW_LIST).strip()
    )
    assert (clone / LIST_PATH).read_bytes() == b"REFORMATTED BY HOOK"
    # `git()` strips, which eats porcelain's leading worktree-column space; a STAGED change
    # would leave two spaces here and still read differently.
    assert git(clone, "status", "--porcelain") == f"M {LIST_PATH}"


def test_the_identity_comes_from_the_clones_own_config(clone):
    """Never `-c user.name` / `-c user.email` on the commit: the clone decides who committed."""
    git(clone, "config", "user.name", "Somebody Local")
    git(clone, "config", "user.email", "local@example.invalid")

    head = emit_branch(
        clone,
        branch="uadclaw/identity",
        base_ref="base",
        list_path=LIST_PATH,
        new_bytes=_bytes(NEW_LIST),
        message="feat(lists): a batch",
    )

    assert git(clone, "log", "-1", "--format=%an <%ae>", head) == (
        "Somebody Local <local@example.invalid>"
    )


# --- emit_branch: what it refuses --------------------------------------------------------------


def test_an_existing_branch_is_refused_and_nothing_moves(clone):
    """A failed emission that left a half-made branch behind would be refused here forever."""
    git(clone, "branch", "uadclaw/taken", "base")
    before = git(clone, "rev-parse", "uadclaw/taken")

    with pytest.raises(UpstreamRepoError, match="already exists"):
        emit_branch(
            clone,
            branch="uadclaw/taken",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert git(clone, "rev-parse", "uadclaw/taken") == before
    assert git(clone, "symbolic-ref", "--short", "HEAD") == "base"


def test_bytes_identical_to_the_base_are_refused_rather_than_committed_empty(clone):
    with pytest.raises(UpstreamRepoError, match="would commit nothing"):
        emit_branch(
            clone,
            branch="uadclaw/no-op",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(BASE_LIST),
            message="feat(lists): a batch",
        )

    assert branch_exists(clone, "uadclaw/no-op") is False


def test_bytes_that_are_not_a_json_object_never_reach_a_branch(clone):
    """Validating only what is read leaves the write side free to put anything on a branch a
    human is about to push."""
    with pytest.raises(UpstreamRepoError, match="not valid JSON"):
        emit_branch(
            clone,
            branch="uadclaw/garbage",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=b"<html>upstream said no</html>",
            message="feat(lists): a batch",
        )

    assert branch_exists(clone, "uadclaw/garbage") is False
    assert (clone / LIST_PATH).read_bytes() == _bytes(BASE_LIST)


def test_an_empty_commit_message_is_refused(clone):
    with pytest.raises(UpstreamRepoError, match="commit message is empty"):
        emit_branch(
            clone,
            branch="uadclaw/no-message",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="   \n",
        )

    assert branch_exists(clone, "uadclaw/no-message") is False


def test_a_dirty_clone_is_refused_before_any_branch_is_created(clone):
    (clone / "half-finished.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(UpstreamRepoError, match="uncommitted changes"):
        emit_branch(
            clone,
            branch="uadclaw/on-top-of-somebody",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert branch_exists(clone, "uadclaw/on-top-of-somebody") is False


# --- emit_branch: rollback ---------------------------------------------------------------------


def test_a_failure_after_the_branch_was_created_leaves_the_clone_exactly_as_it_was(clone):
    """A commit hook refusing is the shape this rollback exists for: the branch was created, the
    checkout moved and the file was written before anything could fail."""
    install_hook(clone, "exit 1")
    before = git(clone, "rev-parse", "HEAD")

    with pytest.raises(UpstreamRepoError, match="commit"):
        emit_branch(
            clone,
            branch="uadclaw/rolled-back",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert git_code(clone, "show-ref", "--verify", "--quiet", "refs/heads/uadclaw/rolled-back") == 1
    assert git(clone, "symbolic-ref", "--short", "HEAD") == "base"
    assert git(clone, "rev-parse", "HEAD") == before
    assert git(clone, "status", "--porcelain") == ""
    assert (clone / LIST_PATH).read_bytes() == _bytes(BASE_LIST)


def test_a_clone_with_no_committer_identity_fails_and_is_restored(clone, monkeypatch):
    """No identity is injected on the commit line, so a clone that configures none cannot
    commit — and the operator is told to configure the clone rather than silently getting a
    machine-shaped author."""
    for leaked in ("EMAIL", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL", "GIT_AUTHOR_NAME"):
        monkeypatch.delenv(leaked, raising=False)
    git(clone, "config", "--unset", "user.email")
    git(clone, "config", "user.useConfigOnly", "true")

    with pytest.raises(UpstreamRepoError, match="exited"):
        emit_branch(
            clone,
            branch="uadclaw/no-identity",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert branch_exists(clone, "uadclaw/no-identity") is False
    assert git(clone, "symbolic-ref", "--short", "HEAD") == "base"
    assert git(clone, "status", "--porcelain") == ""


def test_a_detached_clone_is_restored_detached_at_the_same_commit(clone):
    """The other arm of "where was HEAD": a full `refs/heads/<name>` checkout would detach a
    clone that was on a branch, and a branch name cannot restore one that was not."""
    install_hook(clone, "exit 1")
    git(clone, "checkout", "--quiet", "--detach", "base")
    before = git(clone, "rev-parse", "HEAD")

    with pytest.raises(UpstreamRepoError):
        emit_branch(
            clone,
            branch="uadclaw/detached",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert git(clone, "rev-parse", "HEAD") == before
    assert git_code(clone, "symbolic-ref", "--quiet", "HEAD") != 0
    assert branch_exists(clone, "uadclaw/detached") is False


def test_a_hook_that_rewrites_the_list_is_caught_and_the_branch_discarded(clone):
    """The one failure that would otherwise ship: the commit succeeds, so nothing raises, and a
    reformatted uad_lists.json reaches the PR as a multi-thousand-line diff upstream rejects."""
    install_hook(
        clone,
        f"printf '%s' '{{\"com.rewritten.by.hook\": {{}}}}' > {LIST_PATH}\ngit add -- {LIST_PATH}",
    )
    before = git(clone, "rev-parse", "HEAD")

    with pytest.raises(UpstreamRepoError, match="does not carry the bytes it was given"):
        emit_branch(
            clone,
            branch="uadclaw/rewritten",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert branch_exists(clone, "uadclaw/rewritten") is False
    assert git(clone, "rev-parse", "HEAD") == before
    assert git(clone, "status", "--porcelain") == ""
    assert (clone / LIST_PATH).read_bytes() == _bytes(BASE_LIST)


def test_a_hook_that_adds_a_second_file_is_caught_and_the_branch_discarded(clone):
    """A branch this pipeline emits carries one file. The commit succeeds, so only reading it
    back finds this — and the PR is what upstream reads.

    The hook's own untracked leftover survives the rollback on purpose: `git clean` in an
    operator's clone is not this module's call, and the next run refuses the clone by name.
    """
    install_hook(clone, "printf 'x' > sneaky.txt\ngit add -- sneaky.txt")
    before = git(clone, "rev-parse", "HEAD")

    with pytest.raises(UpstreamRepoError, match="rather than only") as raised:
        emit_branch(
            clone,
            branch="uadclaw/two-files",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert "sneaky.txt" in str(raised.value)
    assert branch_exists(clone, "uadclaw/two-files") is False
    assert git(clone, "rev-parse", "HEAD") == before
    assert (clone / LIST_PATH).read_bytes() == _bytes(BASE_LIST)


def test_a_branch_name_git_resolves_elsewhere_never_commits_onto_the_operators_branch(clone):
    """`@` passes `check-ref-format --branch` AND echoes itself, so the shorthand guard at 170
    lets it through; `git branch @` then creates `refs/heads/@` while `git checkout @` resolves
    `@` as the HEAD SYNONYM and moves nothing. Measured on git 2.55: the write, the add and the
    commit all land on whatever branch the operator was sitting on, and both read-backs pass
    because they only ever describe the commit that was made.

    So the guard cannot be a bigger name blocklist — it has to be "HEAD is attached to the ref
    this emission created", asserted before a single byte is written.
    """
    base_before = git(clone, "rev-parse", "base")

    with pytest.raises(UpstreamRepoError, match="did not check out") as raised:
        emit_branch(
            clone,
            branch="@",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert "'@'" in str(raised.value)
    assert git(clone, "rev-parse", "base") == base_before
    assert git_code(clone, "show-ref", "--verify", "--quiet", "refs/heads/@") == 1
    assert git(clone, "symbolic-ref", "--short", "HEAD") == "base"
    assert git(clone, "status", "--porcelain") == ""
    assert (clone / LIST_PATH).read_bytes() == _bytes(BASE_LIST)


def test_a_commit_that_lands_on_the_branch_mid_emission_is_refused(clone):
    """The two read-backs bound the COMMIT; the artifact a human pushes is the BRANCH. A human
    committing in the clone during the window puts their commit under ours, and every check that
    reads only `HEAD`'s own diff against its parent says the emission was perfect.

    The pin is the parent: exactly one commit on top of `base_commit`, so a second one cannot
    ride along.

    The hook deletes itself as its first act. `post-checkout` fires again during the restore's
    own checkout, and a second firing would commit onto `base` — which would be the probe moving
    the thing the assertion is watching, not the module.
    """
    install_hook(
        clone,
        'rm -f -- "$0"\n'
        "printf 'notes\\n' > README.md\n"
        "git add -- README.md\n"
        "git commit --quiet -m 'human wip'",
        name="post-checkout",
    )
    base_before = git(clone, "rev-parse", "base")

    with pytest.raises(UpstreamRepoError, match="is not the commit the branch was cut from"):
        emit_branch(
            clone,
            branch="uadclaw/ridden",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert branch_exists(clone, "uadclaw/ridden") is False
    assert git(clone, "rev-parse", "base") == base_before


def test_a_humans_in_flight_edit_to_another_file_survives_a_failed_emission(clone):
    """`--force` on the way back would delete work no git object holds: no stash, no reflog, no
    recovery. The emission owns exactly `list_path`, so the restore discards exactly that path
    and then switches back with a PLAIN checkout.

    `pre-commit` rather than `post-checkout` for the writer: `post-checkout` fires a second time
    during the restore's own checkout and re-creates the file, which reads as a pass while
    proving nothing.
    """
    (clone / "README.md").write_text("ORIGINAL COMMITTED\n", encoding="utf-8")
    git(clone, "add", "--", "README.md")
    git(clone, "commit", "--quiet", "-m", "readme")
    before = git(clone, "rev-parse", "HEAD")
    install_hook(clone, "printf 'HUMAN UNSAVED WORK\\n' > README.md\nexit 1")

    with pytest.raises(UpstreamRepoError):
        emit_branch(
            clone,
            branch="uadclaw/keeps-their-work",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    assert (clone / "README.md").read_text() == "HUMAN UNSAVED WORK\n"
    assert (clone / LIST_PATH).read_bytes() == _bytes(BASE_LIST)
    assert git(clone, "symbolic-ref", "--short", "HEAD") == "base"
    assert git(clone, "rev-parse", "HEAD") == before
    assert branch_exists(clone, "uadclaw/keeps-their-work") is False


def test_a_clone_stopped_mid_rebase_is_refused_even_though_its_tree_is_clean(clone):
    """`status --porcelain` is empty at a rebase stop, so the dirty check waves it through and
    the emission's checkout strands the operator's rebase on the emission branch."""
    git(clone, "checkout", "--quiet", "-b", "topic")
    (clone / "topic.txt").write_text("t", encoding="utf-8")
    git(clone, "add", "--", "topic.txt")
    git(clone, "commit", "--quiet", "-m", "topic work")
    assert git_code(clone, "rebase", "--exec", "false", "base") != 0
    assert git(clone, "status", "--porcelain") == ""

    with pytest.raises(UpstreamRepoError, match="has a rebase-merge in progress") as raised:
        inspect_repo(clone, base_ref="base", list_path=LIST_PATH)
    assert "rebase" in str(raised.value)


def test_a_restore_that_only_partly_finished_reports_the_state_it_measured(clone):
    """A stale `.git/index.lock` — what a SIGKILLed git or the live human leaves — fails every
    checkout with 128 while `git branch` and `git branch -D` both succeed without one. The old
    wording named the clone as "left on <branch>" and told the operator to check `base` out by
    hand; both were already true. Nothing may be claimed here that was not read back."""
    lock = clone / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    try:
        with pytest.raises(UpstreamRepoError) as raised:
            emit_branch(
                clone,
                branch="uadclaw/locked",
                base_ref="base",
                list_path=LIST_PATH,
                new_bytes=_bytes(NEW_LIST),
                message="feat(lists): a batch",
            )
    finally:
        lock.unlink(missing_ok=True)

    message = str(raised.value)
    assert git(clone, "symbolic-ref", "--short", "HEAD") == "base"
    assert branch_exists(clone, "uadclaw/locked") is False
    # The positive leg, and the one that discriminates: the restore had nothing to undo because
    # HEAD never moved, so it reports NO problem and the original failure is raised as itself.
    # Without the skip, both checkouts run, both fail on the lock, and the operator is handed a
    # restore failure for work that was never done.
    assert "could not be put back" not in message
    assert "checkout" in message and "exited 128" in message


def test_a_non_ascii_list_path_is_not_read_as_a_rewritten_commit(clone, tmp_path):
    """`diff-tree --name-only` C-quotes a non-ASCII path under the default `core.quotePath`, so
    a correct commit compares unequal to `[list_path]` and is rolled back — after the commit,
    which is the expensive place to be wrong."""
    accented = "resources/assets/uad_lïsts.json"
    git(clone, "mv", LIST_PATH, accented)
    git(clone, "commit", "--quiet", "-m", "rename the list")

    head = emit_branch(
        clone,
        branch="uadclaw/accented",
        base_ref="base",
        list_path=accented,
        new_bytes=_bytes(NEW_LIST),
        message="feat(lists): a batch",
    )

    assert (
        git(clone, "cat-file", "blob", f"{head}:{accented}").encode("utf-8")
        == _bytes(NEW_LIST).strip()
    )


def test_a_restore_that_cannot_finish_names_the_branch_the_clone_is_left_on(clone, monkeypatch):
    """The operator's fix differs from the one the original failure asks for, so it cannot live
    in a log line the dashboard never shows."""
    import uadclaw.upstreamrepo as module

    monkeypatch.setattr(
        module,
        "_restore",
        lambda root, *, original, branch, list_path: (
            f"could not check {original} back out. HEAD is on {branch}, and {branch} still exists"
        ),
    )
    install_hook(clone, "exit 1")

    with pytest.raises(UpstreamRepoError, match="could not be put back") as raised:
        emit_branch(
            clone,
            branch="uadclaw/stuck",
            base_ref="base",
            list_path=LIST_PATH,
            new_bytes=_bytes(NEW_LIST),
            message="feat(lists): a batch",
        )

    message = str(raised.value)
    assert "uadclaw/stuck" in message
    assert "base" in message
    # The wrapper carries the original failure's own text, which already names the operation.
    assert message.count("emit_branch:") == 1
