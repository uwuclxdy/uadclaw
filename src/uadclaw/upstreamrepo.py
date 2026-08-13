"""The operator's local clone of the upstream repo, and the branch the last stage writes into it.

Nothing here pushes, fetches or authenticates, and that is a decision rather than an omission:
there is no GitHub credential anywhere in this stack (design decision 8), so the pipeline's last
act is a local commit on a local branch and a human runs `git push` from the clone afterwards.
Every call is `git` in a list argv with a timeout, never a shell.

**The clone may be a fork**, so nothing here validates it by its remote URL, its remote name or
its default branch name. It is validated by CONTENT: the path is a git work tree, and the
configured list path holds a parseable non-empty JSON object at the base ref. Same idiom as
`unpack.py`'s dispatch-on-the-bytes rule — a name is what an upstream can rename, and a fork
renames all three of those by definition.

`list_bytes` is read with `git cat-file` at the resolved base COMMIT rather than off the working
tree, so what an emission edits is what the branch will actually be based on, and a clone sitting
on some other branch cannot silently contribute its own copy.

What state the clone is left in, per outcome:

- `inspect_repo` and `branch_exists` are read-only. They never move HEAD, never write the index
  and never touch the working tree, so they are safe against a clone a human is sitting in.
- `emit_branch` succeeds: the clone is ON the new branch with the new commit at HEAD, because
  the human's next act is `git push` from there. The working tree is clean unless one of the
  clone's own hooks writes into it AFTER the commit — a `post-commit` formatter does exactly
  that, and the branch is still correct, so the emission reports success and the next one
  refuses the clone until somebody cleans it up.
- `emit_branch` fails, at any step: the emission's own edit to the list file is discarded, the
  ref that was checked out on entry is checked out again, and the created branch is deleted, so
  a failed emission leaves nothing behind for a later run to refuse as already existing. Only
  the list file is discarded — anything else in the clone belongs to a human and is left alone,
  even when leaving it alone is what makes the switch back fail.
- `emit_branch` fails AND the restore fails too: the raised error carries the original failure
  plus what could not be undone, and every claim in it about where HEAD is and whether the
  branch survived is read back rather than assumed.

The clone's own hooks run — no `--no-verify`, since disabling a hook the operator installed is
their decision and not this module's. What a successful commit produced is therefore read back
before the emission reports success, and all three checks are needed because they fail
differently: the committed blob must equal the bytes handed in (a hook that reformats
`uad_lists.json` is a documented upstream rejection reason), the commit must touch that path and
nothing else, and its parent must be the commit the branch was cut from — the first two describe
the COMMIT, while what a human pushes is the BRANCH, and a human committing into the clone
mid-emission puts their work underneath ours where neither of the first two can see it.

An emission also refuses a clone that is stopped inside a rebase, merge, cherry-pick, revert or
bisect. That state is invisible to `status --porcelain` (a rebase stopped at an `edit` step
stages nothing), and cutting a branch under it strands the operator's operation on the emission
branch when they continue it.

Synchronous on purpose, matching the API this module was specified against. The worker runs an
event loop, so a stage calls these through `asyncio.to_thread` rather than inline.
"""

import json
import logging
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

# Every git call gets one. A checkout of this repo is a few MB and every operation here is local,
# so a call still running after this is a hung git (a credential or editor prompt, a wedged hook)
# rather than slow work, and a worker blocked forever on it holds the job and its scratch lease.
#
# It bounds THIS MODULE'S WAIT and nothing else. `subprocess.run(timeout=...)` kills the direct
# child only; measured on git 2.55 with a `pre-commit` sleeping 6s against a 1s deadline, the
# TimeoutExpired surfaced at 1.00s and the hook went on to rewrite the working tree 6.5s later,
# after the emission had given up. Reaping it would need a process group this module does not
# create, and killing an operator's own hooks is a behaviour change nobody has decided on, so
# the timeout says what it leaves behind instead.
GIT_TIMEOUT_SECONDS = 120

# Stripped from the environment every git call inherits. These override `git -C <path>` rather
# than combining with it, so one of them left in a parent process retargets reads and writes at a
# different repository with nothing raised. Not hypothetical: the global pre-commit hook exports
# `GIT_INDEX_FILE` to every child of a `git commit -- <files>`, which is why this repo bans that
# spelling of a commit outright.
_REDIRECTING_GIT_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_NAMESPACE",
)

# A git operation the clone can be stopped in the middle of. Every one of these is invisible to
# `status --porcelain` in at least one of its states — a rebase stopped at an `edit` step stages
# nothing at all — so a clone mid-rebase reads as quiescent and an emission's checkout strands
# that operation on the emission branch.
_INTERRUPTED_OPERATIONS = (
    "rebase-merge",
    "rebase-apply",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
)


class UpstreamRepoError(RuntimeError):
    """The operator's clone cannot be used for an emission, or an emission failed against it.

    Bad operator input rather than a bug here: a missing clone, a dirty work tree, a base ref
    this clone does not carry, a list path that is not the upstream list. Always fatal to the
    emission — a branch is the artifact a human pushes, so a half-made one is worse than none.
    """


@dataclass(frozen=True, slots=True)
class RepoState:
    """The clone as it stood when it was inspected, and the exact bytes a branch would edit."""

    path: str
    base_ref: str
    # The full object id `base_ref` resolved to, so every later call names the commit rather than
    # the ref: a ref can move between the inspection and the commit, an object id cannot.
    base_commit: str
    # Repo-relative and POSIX-spelled, which is the form `git cat-file` and `git add` both take.
    list_path: str
    list_bytes: bytes


def _git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _REDIRECTING_GIT_ENV}


def _failure_tail(result: subprocess.CompletedProcess[bytes]) -> str:
    """git does not put every failure on stderr: a commit with nothing staged exits 1 with
    `nothing to commit` on STDOUT, so a message built from stderr alone reads empty exactly
    where the operator has to read it."""
    streams = (result.stderr, result.stdout)
    lines = [
        stripped
        for stream in streams
        for line in stream.decode("utf-8", "replace").splitlines()
        if (stripped := line.strip())
    ]
    return " | ".join(lines[-5:]) if lines else "(no output)"


def _text(result: subprocess.CompletedProcess[bytes]) -> str:
    return result.stdout.decode("utf-8", "replace").strip()


def _run_git(
    root: Path, argv: Sequence[str], *, context: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    """Run one git command against `root`, raising `UpstreamRepoError` carrying git's own output.

    `check=False` returns the result for a caller that reads the exit code itself (an absent ref
    is an answer, not a failure); every other caller gets the raise.
    """
    command = ["git", "-C", str(root), *argv]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env=_git_env(),
            check=False,
        )
    except FileNotFoundError as exc:
        raise UpstreamRepoError(
            f"{context}: git is not on PATH. Branch emission shells out to it; install git in "
            "the worker image (it is in the runtime stage's apt list and asserted at build time)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpstreamRepoError(
            f"{context}: `{shlex.join(command)}` did not finish within {GIT_TIMEOUT_SECONDS}s. "
            "Every operation here is local, so this is a git waiting on something — a hook, an "
            "editor, or a credential prompt — rather than slow work. Only that git was killed: "
            "anything it spawned is still running and can still write into the clone after this, "
            "and a git killed mid-write leaves `.git/index.lock` behind, which is what the next "
            "emission will refuse on with no other visible cause. Check for both before rerunning."
        ) from exc
    if check and result.returncode != 0:
        raise UpstreamRepoError(
            f"{context}: `{shlex.join(command)}` exited {result.returncode}: "
            f"{_failure_tail(result)}"
        )
    return result


def _validate_branch_name(root: Path, branch: str, *, context: str) -> None:
    """Refuse a branch name git would read as anything other than that literal name.

    Part of the name comes from a vendor string out of the database, so this is untrusted input
    reaching a git ref. Two directions, and only the first is obvious: a name git calls invalid,
    and a name git calls valid but EXPANDS. `git check-ref-format --branch '@{-1}'` prints the
    previously checked-out branch and exits 0 (measured on git 2.55), so requiring its echo to
    equal the input is what separates a literal name from a shorthand.
    """
    if not branch or branch.startswith("-"):
        raise UpstreamRepoError(
            f"{context}: branch name {branch!r} is empty or starts with a dash, which git "
            "reads as an option rather than a name. Name the branch after the vendor batch, "
            "e.g. `uadclaw/samsung-2026-08-13`."
        )
    result = _run_git(root, ["check-ref-format", "--branch", branch], context=context)
    canonical = _text(result)
    if canonical != branch:
        raise UpstreamRepoError(
            f"{context}: branch name {branch!r} is a git shorthand that expands to "
            f"{canonical!r} rather than a literal name. Emit under a name that spells itself."
        )


def _list_path_spec(root: Path, list_path: str, *, context: str) -> tuple[str, Path]:
    """The repo-relative POSIX spelling git is given, and the absolute file that gets written.

    Checked before any git call and before any write, both ends: the configured value could climb
    out of the clone on its own, and the path could BE a symlink out of it — `resolve()` follows
    that, so containment is asserted against the resolved work-tree root rather than against the
    string.
    """
    if not list_path or list_path.startswith("-"):
        raise UpstreamRepoError(
            f"{context}: list_path {list_path!r} is empty or starts with a dash. It names the "
            "upstream list inside the clone, e.g. `resources/assets/uad_lists.json`."
        )
    relative = PurePosixPath(list_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise UpstreamRepoError(
            f"{context}: list_path {list_path!r} is absolute or climbs out of the clone. It "
            "is a path INSIDE the clone, relative to its root."
        )
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise UpstreamRepoError(
            f"{context}: list_path {list_path!r} resolves to {destination}, outside the clone "
            f"at {root}; refusing to read or write it."
        )
    return relative.as_posix(), destination


def _parse_list(raw: bytes, *, context: str, source: str) -> dict[str, Any]:
    """The fork-safe identity check, and the same check on the way back out.

    A clone carrying a parseable non-empty `uad_lists.json` at the configured path IS the right
    repo whoever owns its remote. Applied to the bytes about to be committed too, because
    validating only what is read leaves the write side free to put anything on a branch a human
    is about to push.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UpstreamRepoError(
            f"{context}: {source} is not valid JSON ({exc}). uad_lists.json is a JSON object "
            "keyed by package name; check that list_path names it and not another file."
        ) from exc
    if not isinstance(parsed, dict):
        raise UpstreamRepoError(
            f"{context}: {source} parsed to {type(parsed).__name__}, not an object. "
            "uad_lists.json is a JSON object keyed by package name."
        )
    if not parsed:
        raise UpstreamRepoError(
            f"{context}: {source} carries zero entries. An empty list is not the upstream one, "
            "and a branch built from it would propose deleting every entry."
        )
    return parsed


def _work_tree_root(path: Path, *, context: str) -> Path:
    """The clone's own root, so `<root>/<list_path>` and `git cat-file <rev>:<list_path>` cannot
    disagree — the second is always resolved from the work tree root, whatever directory inside
    the clone the operator configured."""
    if not path.is_dir():
        raise UpstreamRepoError(
            f"{context}: {path} is not a directory. Branch emission needs a local clone of the "
            "upstream repo (a fork is fine) on disk; clone one there and point the setting at it."
        )
    inside = _run_git(path, ["rev-parse", "--is-inside-work-tree"], context=context, check=False)
    if inside.returncode != 0 or _text(inside) != "true":
        raise UpstreamRepoError(
            f"{context}: {path} is not a git work tree ({_failure_tail(inside)}). Emission "
            "commits a branch there, so it has to be a clone rather than a copy of the files."
        )
    return Path(_text(_run_git(path, ["rev-parse", "--show-toplevel"], context=context)))


def _interrupted_operation(root: Path, *, context: str) -> str | None:
    """The git operation this clone is stopped in the middle of, if any.

    Every path is resolved with `--git-path` rather than by joining `.git/` onto the work tree
    root, because a linked worktree keeps these files somewhere else entirely and this module
    otherwise supports one.
    """
    argv = ["rev-parse"]
    for name in _INTERRUPTED_OPERATIONS:
        argv += ["--git-path", name]
    resolved = _text(_run_git(root, argv, context=context)).splitlines()
    for name, path in zip(_INTERRUPTED_OPERATIONS, resolved, strict=True):
        if (root / path).exists():
            return name
    return None


def _head_position(root: Path) -> str | None:
    """Where HEAD is, spelled the way `_restore` has to check it back out, or None when git
    cannot say.

    The SHORT branch name, never `refs/heads/<name>`: checking out the full ref name detaches
    HEAD (measured on git 2.55), which would put the operator's clone back at the right commit
    and off its branch. A clone that was already detached is restored to its commit.
    """
    symbolic = _run_git(
        root, ["symbolic-ref", "--short", "--quiet", "HEAD"], context="emit_branch", check=False
    )
    if symbolic.returncode == 0:
        return _text(symbolic)
    detached = _run_git(root, ["rev-parse", "HEAD"], context="emit_branch", check=False)
    return _text(detached) if detached.returncode == 0 else None


def _branch_ref_state(root: Path, branch: str, *, context: str) -> bool | None:
    """Whether `refs/heads/<branch>` is there. None when git answered neither yes nor no, which
    is not the same as "no" anywhere it gets reported to an operator."""
    result = _run_git(
        root,
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        context=context,
        check=False,
    )
    if result.returncode not in (0, 1):
        return None
    return result.returncode == 0


def _restore(root: Path, *, original: str, branch: str, list_path: str) -> str | None:
    """Undo a failed emission, returning what could not be undone rather than raising.

    Surgical rather than forced, and the difference is somebody's unsaved work. This emission
    owns exactly `list_path`, so exactly that path is discarded and the switch back is a PLAIN
    checkout. A forced switch would delete a human's in-flight edit to any other file in the
    clone, and nothing holds it — no stash, no reflog, no object ever written. A plain checkout
    that refuses because their edit really does conflict with the target is the CORRECT outcome:
    it is reported here rather than overridden.

    The discard has to come first: measured on git 2.55, a staged edit survives a plain switch
    whenever the file matches on both sides, so this emission's own edit would otherwise ride
    back onto the operator's branch. It needs no `--force` — the pathspec form of `checkout`
    overwrites index and working tree without one, which is the same footgun that makes
    `git checkout <ref> -- <file>` lose an edit in everyday use.

    Both checkouts are skipped when HEAD never moved. The emission writes nothing before its own
    checkout, so there is nothing to undo, and a step that was never needed must not report a
    failure the operator then goes looking for.

    Deleting the branch orphans anything else that was committed onto it during the emission.
    That is recoverable — the reflog holds it until gc — which is what separates it from an
    unsaved edit, and is why the branch goes and the working tree stays.
    """
    problems: list[str] = []

    def attempt(argv: list[str]) -> None:
        try:
            _run_git(root, argv, context="_restore")
        except UpstreamRepoError as exc:
            logger.exception("_restore: %s failed in %s", argv[0], root)
            problems.append(str(exc))

    if _head_position(root) != original:
        attempt(["checkout", original, "--", list_path])
        attempt(["checkout", "--quiet", original, "--"])
    attempt(["branch", "-D", branch])
    if not problems:
        return None

    head = _head_position(root) or "an unreadable HEAD"
    remains = _branch_ref_state(root, branch, context="_restore")
    verdict = {True: "still exists", False: "was deleted"}.get(remains, "could not be read")
    return f"{'; '.join(problems)}. HEAD is on {head}, and {branch} {verdict}"


def inspect_repo(path: Path, *, base_ref: str, list_path: str) -> RepoState:
    """Describe the operator's clone, or raise `UpstreamRepoError` naming what to fix.

    Read-only: nothing here moves HEAD, writes the index or touches the working tree.
    """
    root = _work_tree_root(path, context="inspect_repo")
    posix_list_path, _ = _list_path_spec(root, list_path, context="inspect_repo")

    # Before the dirty check, not after: a conflicted rebase is dirty too, and "you have
    # uncommitted changes" would send the operator to fix the wrong thing.
    interrupted = _interrupted_operation(root, context="inspect_repo")
    if interrupted is not None:
        raise UpstreamRepoError(
            f"inspect_repo: the clone at {root} has a {interrupted} in progress. Its work tree "
            "can be perfectly clean mid-rebase, and cutting a branch here would strand that "
            "operation on the emission branch when it is continued. Finish or abort it "
            "(`git rebase --abort`, `git merge --abort`, ...) first."
        )

    # `--no-optional-locks` is what makes this actually read-only. Measured on git 2.55 over 200
    # files touched without changing their bytes: a plain `status --porcelain` refreshes the stale
    # stat-cache and REWRITES `.git/index`, and with the flag it does not. That is a write into a
    # clone a human may be running their own git in at that moment.
    status = _run_git(
        root, ["--no-optional-locks", "status", "--porcelain"], context="inspect_repo"
    )
    if _text(status):
        first = " | ".join(_text(status).splitlines()[:5])
        raise UpstreamRepoError(
            f"inspect_repo: the clone at {root} has uncommitted changes ({first}). A branch cut "
            "here would carry somebody else's half-finished work into the PR, so commit, discard "
            "or stash them first. Untracked files count."
        )

    if not base_ref or base_ref.startswith("-"):
        raise UpstreamRepoError(
            f"inspect_repo: base_ref {base_ref!r} is empty or starts with a dash, which git "
            "reads as an option rather than a ref."
        )
    revision = _run_git(
        root,
        ["rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}"],
        context="inspect_repo",
        check=False,
    )
    if revision.returncode != 0:
        raise UpstreamRepoError(
            f"inspect_repo: base_ref {base_ref!r} does not resolve to a commit in {root} "
            f"({_failure_tail(revision)}). A fork can name its default branch anything, so name "
            "a ref this clone actually carries and fetch it if it is behind."
        )
    base_commit = _text(revision)

    spec = f"{base_commit}:{posix_list_path}"
    kind = _run_git(root, ["cat-file", "-t", spec], context="inspect_repo", check=False)
    if kind.returncode != 0:
        raise UpstreamRepoError(
            f"inspect_repo: {posix_list_path!r} does not exist at {base_ref} in {root} "
            f"({_failure_tail(kind)}). That path is what identifies this clone as the upstream "
            "repo; check the setting against the clone's own layout."
        )
    if _text(kind) != "blob":
        raise UpstreamRepoError(
            f"inspect_repo: {posix_list_path!r} at {base_ref} is a {_text(kind)}, not a file. "
            "list_path names uad_lists.json itself, not the directory holding it."
        )
    list_bytes = _run_git(root, ["cat-file", "blob", spec], context="inspect_repo").stdout
    entries = _parse_list(
        list_bytes, context="inspect_repo", source=f"{posix_list_path} at {base_ref}"
    )

    logger.info(
        "upstream clone %s: %s at %s (%s) carries %d entries",
        root,
        posix_list_path,
        base_ref,
        base_commit[:12],
        len(entries),
    )
    return RepoState(
        path=str(root),
        base_ref=base_ref,
        base_commit=base_commit,
        list_path=posix_list_path,
        list_bytes=list_bytes,
    )


def branch_exists(path: Path, branch: str) -> bool:
    """Whether the clone already carries this branch. Read-only, like `inspect_repo`."""
    root = _work_tree_root(path, context="branch_exists")
    _validate_branch_name(root, branch, context="branch_exists")
    state = _branch_ref_state(root, branch, context="branch_exists")
    if state is None:
        raise UpstreamRepoError(
            f"branch_exists: could not read refs/heads/{branch} in {root}. An emission decides "
            "whether to run on this answer, so an unreadable ref is refused rather than read as "
            "absent."
        )
    return state


def emit_branch(
    path: Path,
    *,
    branch: str,
    base_ref: str,
    list_path: str,
    new_bytes: bytes,
    message: str,
) -> str:
    """Commit `new_bytes` as the list file on a new `branch` cut from `base_ref`, returning the
    commit's full object id and leaving the clone on that branch for the human to push.

    Every failure puts the clone back where it was found; see the module docstring for what each
    outcome leaves behind.
    """
    state = inspect_repo(path, base_ref=base_ref, list_path=list_path)
    root = Path(state.path)

    _validate_branch_name(root, branch, context="emit_branch")
    if branch_exists(root, branch):
        raise UpstreamRepoError(
            f"emit_branch: branch {branch!r} already exists in {root}. A previous emission wrote "
            "it: push and delete it, or emit this batch under another name. Nothing was changed."
        )
    if not message.strip():
        raise UpstreamRepoError(
            "emit_branch: the commit message is empty. It is what a reviewer reads first on the "
            "PR, so it is required rather than defaulted."
        )
    _parse_list(new_bytes, context="emit_branch", source="the list this emission would commit")
    if new_bytes == state.list_bytes:
        raise UpstreamRepoError(
            f"emit_branch: the bytes handed in are identical to {state.list_path} at {base_ref}, "
            "so this would commit nothing. An emission with no additions is a bug upstream of "
            "here, not an empty commit to make."
        )

    original = _head_position(root)
    if original is None:
        raise UpstreamRepoError(
            f"emit_branch: cannot read where HEAD is in {root}, so there is nothing to put the "
            "clone back to if this fails. Refusing to move a checkout this cannot restore."
        )
    _run_git(root, ["branch", branch, state.base_commit], context="emit_branch")
    try:
        _run_git(root, ["checkout", "--quiet", branch, "--"], context="emit_branch")
        # The one thing `git checkout <name>` does not promise is that it checked out THAT name.
        # `@` passes every name check and is git's own synonym for HEAD, so the checkout succeeds,
        # moves nothing, and the whole emission lands on the operator's own branch with both
        # read-backs passing (measured on git 2.55). Asserted here because it is the last moment
        # nothing has been written yet, and because a name blocklist only ever knows the names
        # somebody already found.
        attached = _run_git(
            root, ["symbolic-ref", "--quiet", "HEAD"], context="emit_branch", check=False
        )
        if _text(attached) != f"refs/heads/{branch}":
            raise UpstreamRepoError(
                f"emit_branch: checking out {branch!r} did not check out refs/heads/{branch} — "
                f"HEAD is on {_text(attached) or 'a detached commit'}. git read that name as "
                "something other than the branch just created, so the emission would have "
                "committed onto whatever the operator was sitting on. Nothing was written."
            )
        # Resolved after the checkout rather than before it, because the checkout is what decides
        # which bytes sit at that path: containment is asserted against the tree being written.
        destination = _list_path_spec(root, state.list_path, context="emit_branch")[1]
        destination.write_bytes(new_bytes)
        # One path, never `-A` and never `.`: the emission owns exactly this file, and a clone is
        # a directory a human also works in.
        _run_git(root, ["add", "--", state.list_path], context="emit_branch")
        # No `-c user.name` / `-c user.email`, ever. The clone's own config is the identity; a
        # clone with none fails here and the error tells the operator to configure it.
        _run_git(root, ["commit", "--quiet", "-m", message], context="emit_branch")
        head = _text(_run_git(root, ["rev-parse", "HEAD"], context="emit_branch"))
        committed = _run_git(
            root, ["cat-file", "blob", f"{head}:{state.list_path}"], context="emit_branch"
        ).stdout
        if committed != new_bytes:
            raise UpstreamRepoError(
                f"emit_branch: the commit on {branch} does not carry the bytes it was given "
                f"({len(committed)} bytes committed against {len(new_bytes)} handed in). "
                f"Something in {root} — a commit hook, or another process writing to this clone "
                f"— changed {state.list_path} between the write and the commit; upstream rejects "
                "a reformat of that file, so the branch is discarded rather than pushed."
            )
        # The commit is read back rather than assumed, because a commit that SUCCEEDED is the
        # failure nothing else here can see: a hook in the operator's clone can edit the index
        # between `add` and `commit`, and the resulting PR is the artifact upstream reads.
        #
        # `-z` because `--name-only` C-quotes any path outside plain ASCII under the default
        # `core.quotePath` (measured: `resources/assets/uad_lïsts.json` comes back as
        # `"resources/assets/uad_l\303\257sts.json"`), which would fail a correct commit after
        # making it.
        touched = [
            path
            for path in _run_git(
                root,
                ["diff-tree", "--no-commit-id", "--name-only", "-r", "-z", head],
                context="emit_branch",
            )
            .stdout.decode("utf-8", "replace")
            .split("\0")
            if path
        ]
        if touched != [state.list_path]:
            raise UpstreamRepoError(
                f"emit_branch: the commit on {branch} touches {touched} rather than only "
                f"{state.list_path}. A branch this pipeline emits carries one file and nothing "
                f"else; something in {root} added to the index. The branch is discarded."
            )
        # Both checks above describe the COMMIT. What a human pushes is the BRANCH, and a human
        # committing in this clone during the emission puts their commit underneath ours, where
        # every check that reads only HEAD's own diff calls the emission perfect. The parent is
        # what bounds the branch: exactly one commit, on exactly the commit it was cut from.
        lineage = _text(
            _run_git(root, ["rev-list", "--parents", "-n", "1", head], context="emit_branch")
        ).split()
        if lineage[1:] != [state.base_commit]:
            raise UpstreamRepoError(
                f"emit_branch: the commit on {branch} sits on {lineage[1:] or 'no parent'}, which "
                f"is not the commit the branch was cut from ({state.base_commit}). Something else "
                "committed into this clone during the emission and would be pushed as part of "
                "this batch. The branch is discarded; `git reflog` in that clone still holds "
                "whatever was committed onto it."
            )
    except BaseException as exc:
        restore_failure = _restore(
            root, original=original, branch=branch, list_path=state.list_path
        )
        if restore_failure is None:
            raise
        if not isinstance(exc, Exception):
            # An interrupt is never converted into a domain error: the caller asked to stop.
            logger.error("emit_branch: %s could not be put back: %s", root, restore_failure)
            raise
        # `exc` already carries the operation name when it came from here; anything else gets one.
        detail = (
            str(exc)
            if isinstance(exc, UpstreamRepoError)
            else f"emit_branch: {type(exc).__name__}: {exc}"
        )
        raise UpstreamRepoError(
            f"{detail} — and the clone could not be put back: {restore_failure}. Put it right by "
            f"hand ({original} is where it started) before running another emission."
        ) from exc

    logger.info(
        "emitted %s at %s in %s (%d bytes on %s, based on %s)",
        branch,
        head[:12],
        root,
        len(new_bytes),
        state.list_path,
        state.base_commit[:12],
    )
    return head
