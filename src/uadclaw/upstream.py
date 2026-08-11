"""The upstream `uad_lists.json`, loaded from an explicitly configured path.

This file decides what the pipeline proposes: a package already in it is out of the additions
queue, so an empty or stale copy is not a degraded input, it is a wrong one. Read "nothing is
already upstream" back through the filter and the entire corpus becomes a candidate — 393
packages of noise into a queue whose whole point is that a human walks it.

Three consequences shape this module:

- **The source is configuration, never a fetch.** `UPSTREAM_LIST_PATH` names a file the
  operator put there; nothing here reaches the network. A stage that silently re-downloaded
  its own filter input would change what reaches triage between two runs of the same corpus,
  with nothing recorded about why, and this repo's reproducibility anchor is the pinned
  evidence rather than "whatever main said today".
- **Provenance travels with the verdict.** `sha256` and `obtained_at` (the file's own mtime,
  not now) are recorded on every row the filter writes, because "it was already upstream" is
  a claim about a specific 1.6 MB of JSON.
- **Nothing silently proceeds.** A missing, unreadable, malformed, wrongly-shaped or empty
  list raises. There is no "assume nothing is upstream" path, by construction.

Refresh it deliberately, from the upstream repo's `main`:

    curl -fsSL -o data/uad_lists.json \\
      https://raw.githubusercontent.com/Universal-Debloater-Alliance/\\
      universal-android-debloater-next-generation/main/resources/assets/uad_lists.json
"""

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A ceiling on what will be read into memory. The live file is ~1.6 MB over 5372 entries, so
# this is two orders of magnitude of headroom and still refuses to read a mistaken mount
# (a firmware image, a log) into RAM.
MAX_UPSTREAM_LIST_BYTES = 256 * 1024**2


class UpstreamListError(RuntimeError):
    """The upstream list could not be loaded, or loaded to something that cannot be trusted
    as one. Bad operator input rather than a bug here, and always fatal to the filter: the
    fallback ("treat nothing as upstream") is worse than stopping."""


@dataclass(frozen=True, slots=True)
class UpstreamEntry:
    """One existing entry, reduced to the three fields a style anchor shows the model.

    Not a schema for the file: the live copy carries `dependencies`, `neededBy`, `labels` and
    three keys the upstream struct does not even declare (`leabel`, `labelid`, `suggestions`).
    Only what a proposal is written against is kept, because everything else is either
    graph-derived here or noise in a prompt.
    """

    package: str
    list: str | None
    removal: str | None
    description: str


@dataclass(frozen=True, slots=True)
class UpstreamList:
    """The package names already carried upstream, plus which bytes said so."""

    path: str
    sha256: str
    # When this copy was obtained, i.e. the file's mtime. Deliberately not "now": the answer
    # to "how stale is the list this verdict came from" is a property of the file.
    obtained_at: datetime
    loaded_at: datetime
    entry_count: int
    packages: frozenset[str]
    # Package -> the entry's proposal fields, for the style anchors the classification
    # bundle carries. The filter never reads this; it only ever asks `package in upstream`.
    entries: Mapping[str, UpstreamEntry] = field(default_factory=dict)

    def __contains__(self, package: str) -> bool:
        return package in self.packages

    def provenance(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "obtained_at": self.obtained_at.isoformat(),
            "loaded_at": self.loaded_at.isoformat(),
            "entry_count": self.entry_count,
        }


def _entries(parsed: dict[str, Any]) -> dict[str, UpstreamEntry]:
    """Reduce the parsed file to the anchor fields, skipping anything malformed.

    Skipping rather than raising, unlike everything else in this module: a single entry whose
    `list` is a number is a defect in somebody else's file and costs one style anchor, while
    the filter's question ("is this package carried upstream") is still answered correctly by
    its key. The file as a whole is already refused when it is empty or not an object.
    """
    entries: dict[str, UpstreamEntry] = {}
    for package, value in parsed.items():
        if not isinstance(value, dict):
            continue
        description = value.get("description")
        entries[str(package)] = UpstreamEntry(
            package=str(package),
            list=value["list"] if isinstance(value.get("list"), str) else None,
            removal=value["removal"] if isinstance(value.get("removal"), str) else None,
            description=description if isinstance(description, str) else "",
        )
    return entries


def load_upstream_list(path: Path) -> UpstreamList:
    """Read `uad_lists.json` off disk, or raise `UpstreamListError` saying what to fix."""
    if not path.is_file():
        raise UpstreamListError(
            f"load_upstream_list: {path} is not a file. The filter stage needs the upstream "
            "uad_lists.json to know which packages are already carried; point "
            "UPSTREAM_LIST_PATH at a copy (the worker container mounts ./data read-only at "
            "/data) rather than letting the pipeline treat the whole corpus as new."
        )
    size = path.stat().st_size
    if size > MAX_UPSTREAM_LIST_BYTES:
        raise UpstreamListError(
            f"load_upstream_list: {path} is {size} bytes, over the "
            f"{MAX_UPSTREAM_LIST_BYTES}-byte ceiling. That is not uad_lists.json (~1.6 MB); "
            "check what UPSTREAM_LIST_PATH is pointing at."
        )

    raw = path.read_bytes()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UpstreamListError(
            f"load_upstream_list: {path} is not valid JSON ({exc}). A truncated download is "
            "the usual cause; re-fetch it."
        ) from exc
    if not isinstance(parsed, dict):
        raise UpstreamListError(
            f"load_upstream_list: {path} parsed to {type(parsed).__name__}, not an object. "
            "uad_lists.json is a JSON object keyed by package name."
        )

    packages = frozenset(str(key) for key in parsed)
    if not packages:
        raise UpstreamListError(
            f"load_upstream_list: {path} carries zero entries. An empty list would mark every "
            "package in the corpus as missing from upstream and push the whole corpus into "
            "the additions queue, so it is refused rather than used."
        )

    upstream = UpstreamList(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        obtained_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        loaded_at=datetime.now(UTC),
        entry_count=len(packages),
        packages=packages,
        entries=_entries(parsed),
    )
    logger.info(
        "upstream list loaded: %d entries from %s (sha256 %s, obtained %s)",
        upstream.entry_count,
        upstream.path,
        upstream.sha256[:12],
        upstream.obtained_at.isoformat(),
    )
    return upstream
