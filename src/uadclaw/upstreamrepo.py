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
- `emit_branch` succeeds: the clone is ON the new branch with a clean tree and the new commit at
  HEAD, because the human's next act is `git push` from there.
- `emit_branch` fails, at any step: the ref that was checked out on entry is checked out again
  and the created branch is deleted, so a failed emission leaves nothing behind for a later run
  to refuse as already existing.
- `emit_branch` fails AND the restore fails too: the raised error says so and names the branch
  the clone is left on, because that is a different fix (a hand checkout) from the one the
  original failure asks for.

The clone's own hooks run — no `--no-verify`, since disabling a hook the operator installed is
their decision and not this module's. A hook that REWRITES `uad_lists.json` is caught instead:
the committed blob is read back and compared against the bytes handed in, because a reformat of
that file is a documented upstream rejection reason and would otherwise ship inside a content PR.

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
            "editor, or a credential prompt — rather than slow work."
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


def _current_head(root: Path) -> str:
    """What `_restore` has to check out to put the clone back.

    The SHORT branch name, never `refs/heads/<name>`: checking out the full ref name detaches
    HEAD (measured on git 2.55), which would put the operator's clone back at the right commit
    and off its branch. A clone that was already detached is restored to its commit.
    """
    symbolic = _run_git(
        root, ["symbolic-ref", "--short", "--quiet", "HEAD"], context="emit_branch", check=False
    )
    if symbolic.returncode == 0:
        return _text(symbolic)
    return _text(_run_git(root, ["rev-parse", "HEAD"], context="emit_branch"))


def _restore(root: Path, *, original: str, branch: str) -> str | None:
    """Undo a failed emission, returning what could not be undone rather than raising.

    `--force` is correct here and only here: `inspect_repo` refused a dirty tree on entry, so the
    only changes this can discard are the ones the failed emission itself made. A plain checkout
    would CARRY them onto the original branch — measured on git 2.55, a staged modification
    survives the switch whenever the file matches on both sides — leaving the operator's clone
    dirty and the next run refusing it for a mess this one made.
    """
    problems: list[str] = []
    try:
        _run_git(root, ["checkout", "--force", "--quiet", original, "--"], context="_restore")
    except UpstreamRepoError as exc:
        logger.exception("_restore: could not check %s back out in %s", original, root)
        problems.append(str(exc))
    try:
        _run_git(root, ["branch", "-D", branch], context="_restore")
    except UpstreamRepoError as exc:
        logger.exception("_restore: could not delete the half-made branch %s in %s", branch, root)
        problems.append(str(exc))
    return "; ".join(problems) if problems else None


def inspect_repo(path: Path, *, base_ref: str, list_path: str) -> RepoState:
    """Describe the operator's clone, or raise `UpstreamRepoError` naming what to fix.

    Read-only: nothing here moves HEAD, writes the index or touches the working tree.
    """
    root = _work_tree_root(path, context="inspect_repo")
    posix_list_path, _ = _list_path_spec(root, list_path, context="inspect_repo")

    # `--no-optional-locks` is what makes this actually read-only: a plain `git status` refreshes
    # the index and takes `.git/index.lock` to do it, which is a write into a clone a human may be
    # running their own git in at that moment.
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
    result = _run_git(
        root,
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        check=False,
        context="branch_exists",
    )
    if result.returncode not in (0, 1):
        raise UpstreamRepoError(
            f"branch_exists: could not read refs/heads/{branch} in {root} "
            f"({_failure_tail(result)})."
        )
    return result.returncode == 0


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

    original = _current_head(root)
    _run_git(root, ["branch", branch, state.base_commit], context="emit_branch")
    try:
        _run_git(root, ["checkout", "--quiet", branch, "--"], context="emit_branch")
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
                f"({len(committed)} bytes committed against {len(new_bytes)} handed in). A hook "
                f"in {root} rewrote {state.list_path}; upstream rejects a reformat of that file, "
                "so the branch is discarded rather than pushed."
            )
        # The commit is read back rather than assumed, because a commit that SUCCEEDED is the
        # failure nothing else here can see: a hook in the operator's clone can edit the index
        # between `add` and `commit`, and the resulting PR is the artifact upstream reads.
        touched = _text(
            _run_git(
                root,
                ["diff-tree", "--no-commit-id", "--name-only", "-r", head],
                context="emit_branch",
            )
        ).splitlines()
        if touched != [state.list_path]:
            raise UpstreamRepoError(
                f"emit_branch: the commit on {branch} touches {touched} rather than only "
                f"{state.list_path}. A branch this pipeline emits carries one file and nothing "
                f"else; something in {root} added to the index. The branch is discarded."
            )
    except BaseException as exc:
        restore_failure = _restore(root, original=original, branch=branch)
        if restore_failure is None:
            raise
        if not isinstance(exc, Exception):
            # An interrupt is never converted into a domain error: the caller asked to stop.
            logger.error("emit_branch: %s is left on %s: %s", root, branch, restore_failure)
            raise
        raise UpstreamRepoError(
            f"emit_branch: {exc} — and the clone could not be put back: {restore_failure}. "
            f"{root} is left on {branch}; check {original} out by hand and delete that branch "
            "before running another emission."
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
