"""The per-package evidence bundle the model is shown (`docs/pipeline-design.md` §7).

**This is the project's reproducibility anchor.** DeepSeek has no seed and no determinism
guarantee — not even on a cache hit, since caching reuses the input prefix's compute and not
the sampling — so nothing about the model's answer can be pinned. What CAN be pinned is
exactly what it was asked, and that is this bundle: a canonical serialization (sorted keys,
no whitespace variance, no timestamps, no run-scoped ids) plus its sha256. Two runs over the
same corpus produce byte-identical bundles and therefore the same hash, whatever order the
packages arrive in, and that hash is what the upstream PR body links to as the disclosure
CONTRIBUTING.md demands.

Three consequences shape the module:

- **Pure.** No DB, no network, no clock — the same posture as `facts.py` and `ladder.py`, so
  it is exercised by fast tests over synthetic packages rather than only by a corpus. A
  timestamp anywhere in here would silently make every hash unique and the anchor worthless.
- **Everything is sorted.** Not "usually emitted in a stable order": every list is sorted on
  the way in, because the corpus arrives from a database query whose order is only as stable
  as its ORDER BY, and the anchors are chosen out of a 5372-entry mapping.
- **The floor travels as data the prompt shows the model; `dependencies` and `neededBy` do
  not travel as anything the model may edit.** They are in the bundle as evidence — the model
  needs to know what breaks — and the validator refuses a response that carries either key at
  all, so a wrong edge cannot arrive from a model even by accident.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from uadclaw.corpus import CorpusGraph, PackageEvidence
from uadclaw.corpus import CorpusPackage as CorpusPackage
from uadclaw.ladder import RemovalFloor
from uadclaw.upstream import UpstreamEntry, UpstreamList

# Bumped whenever the payload's SHAPE changes. Part of the hashed content on purpose: two
# bundles built by two versions of this code describe the same package differently, and a
# shared hash would claim they are the same evidence.
BUNDLE_SCHEMA_VERSION = 1

# How many existing upstream entries travel as style anchors. Four is a compromise measured
# against the token budget rather than a round number: the live entries have a median
# description of 80 characters, so four anchors cost a few hundred tokens of the per-package
# half of the prompt, and fewer than three stops showing the model that entries differ in
# length and register.
DEFAULT_ANCHOR_COUNT = 4

# How many leading dotted segments an entry must share before it is an anchor at all. Two,
# because one buys nothing: measured against the real list, `com.zzz.vendor.widget` with no
# minimum draws `com.LocalFota`, `com.LogiaGroup.LogiaDeck`, `com.Qunar`, `com.Rogers…` —
# alphabetical noise that shares only `com`, presented to the model as entries to match the
# register of. It also churns the hash: anchors are hashed, so ONE new upstream entry sorting
# ahead of those four moves `bundle_sha256` for every namespace-less package, and the next
# run re-asks and re-bills all of them for evidence that did not change.
MIN_ANCHOR_SHARED_SEGMENTS = 2


@dataclass(frozen=True, slots=True)
class PackageIdentity:
    """The identity fields a bundle shows and the deterministic core has no use for.

    `CorpusPackage` is deliberately the *core's* view — the graph, the ladder and the filter
    never read a label or a certificate — so rather than widening a shared type that three
    pure modules already consume, the extra fields the prompt needs travel beside it.
    """

    label: str = "unknown"
    label_unresolved: bool = False
    # The best `list`-category signal there is: the signing organization survives APK
    # signature v3 key rotation even when the common name does not.
    cert_issuer: str | None = None
    version_code: int | None = None
    # Set when devices disagreed about this package's identity fields. A review signal, not a
    # verdict — 134 of the 147 packages shared by two real corpora carry it, all ordinary key
    # rotation — so it is shown and never used to suppress anything.
    has_conflict: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """One package's evidence, canonically serialized and content-addressed."""

    package: str
    payload: Mapping[str, Any]
    sha256: str

    def canonical_json(self) -> str:
        return canonical_json(self.payload)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """The one serialization a bundle hash is ever taken over.

    `sort_keys` plus the tightest separators: two dicts built in different key orders must
    produce the same bytes, and a stray space would change the digest without changing the
    evidence. `ensure_ascii=False` so a non-ASCII label is one character rather than an
    escape — the digest is over UTF-8 bytes either way, and the prompt reads better.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def bundle_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _shared_prefix_length(left: str, right: str) -> int:
    """How many leading dotted segments two package names share."""
    left_parts = left.split(".")
    right_parts = right.split(".")
    shared = 0
    for a, b in zip(left_parts, right_parts, strict=False):
        if a != b:
            break
        shared += 1
    return shared


def nearest_entries(
    package: str, upstream: UpstreamList, *, limit: int = DEFAULT_ANCHOR_COUNT
) -> tuple[UpstreamEntry, ...]:
    """The existing entries that read most like neighbours of `package`, deterministically.

    The rule is **longest shared dotted prefix, ties broken by package name ascending**, and
    both halves are load-bearing. Dotted prefix rather than an edit distance because a package
    name is a namespace and not a word: `com.motorola.ccc.ota` is a sibling of
    `com.motorola.ccc.notification` in a way `com.motorola.ccc.ota` and `com.motorola.cccota`
    are not, and an edit distance ranks the second pair higher. Name-ascending rather than
    "whichever the mapping yields first" because a bundle whose anchors depend on dict
    iteration order is a bundle whose hash is not reproducible, which is the one property this
    module exists to provide.

    An entry for `package` itself is excluded: a package already carried upstream is being
    re-proposed, and handing the model the answer it is being asked for is not an anchor.

    A package with no namespace neighbour gets NO anchors rather than the alphabetically
    nearest strangers — see `MIN_ANCHOR_SHARED_SEGMENTS`.
    """
    scored = sorted(
        (
            (-shared, entry.package)
            for entry in upstream.entries.values()
            if entry.package != package
            and entry.description.strip()
            and (shared := _shared_prefix_length(package, entry.package))
            >= MIN_ANCHOR_SHARED_SEGMENTS
        ),
    )
    return tuple(upstream.entries[name] for _, name in scored[:limit])


def _anchor_json(entry: UpstreamEntry) -> dict[str, Any]:
    return {
        "package": entry.package,
        "list": entry.list,
        "removal": entry.removal,
        "description": entry.description,
    }


def _facts_json(item: CorpusPackage, identity: PackageIdentity) -> dict[str, Any]:
    return {
        "label": identity.label,
        "label_unresolved": identity.label_unresolved,
        "version_code": identity.version_code,
        "cert_issuer": identity.cert_issuer,
        "partitions": sorted(item.partitions),
        "priv_app": item.priv_app,
        "core_app": item.core_app,
        "persistent": item.persistent,
        "shared_user_id": item.shared_user_id,
        "overlay_target": item.overlay_target,
        "declares_libraries": sorted({*item.libraries, *item.static_libraries}),
        "uses_libraries_required": sorted(item.uses_libraries_required),
        "uses_libraries_optional": sorted(item.uses_libraries_optional),
        "provider_authorities": sorted(item.provider_authorities),
        "handles_home": item.handles_home,
        "handles_dialer": item.handles_dialer,
        "handles_sms_deliver": item.handles_sms_deliver,
        "is_input_method": item.is_input_method,
        "has_conflict": identity.has_conflict,
    }


def _provenance_json(item: CorpusPackage) -> dict[str, Any]:
    """Which devices shipped this package. Device KEYS (`<driver>:<device>`) and never a
    build id: a build id would change every time a vendor pushes an update and would rewrite
    the hash of evidence that did not change."""
    return {
        "devices": sorted(item.devices),
        "device_count": item.device_count,
    }


def _graph_json(graph: CorpusGraph | None, package: str) -> dict[str, Any]:
    if graph is None:
        return {"dependencies": [], "needed_by": [], "edges": [], "evidence": {}}
    evidence = graph.evidence.get(package)
    return {
        "dependencies": sorted(graph.dependencies_of(package)),
        "needed_by": sorted(graph.needed_by(package)),
        "edges": sorted(
            (edge.as_json() for edge in graph.edges_for(package)),
            key=lambda edge: (edge["kind"], edge["dependent"], edge["provider"], edge["detail"]),
        ),
        "evidence": evidence.as_json() if isinstance(evidence, PackageEvidence) else {},
    }


def build_bundle(
    item: CorpusPackage,
    *,
    floor: RemovalFloor,
    identity: PackageIdentity | None = None,
    graph: CorpusGraph | None = None,
    anchors: Sequence[UpstreamEntry] = (),
) -> EvidenceBundle:
    """Everything known about one package, in the one shape its hash is taken over.

    `floor` is required rather than optional: a bundle without it would let a prompt be built
    that never told the model the lower bound it must not go under, and the validator would
    then be rejecting answers to a question nobody asked.
    """
    if floor.package != item.package:
        raise ValueError(
            f"build_bundle: floor is for {floor.package!r} but the package is {item.package!r}. "
            "A bundle carrying another package's floor would show the model the wrong lower "
            "bound and make every rejection meaningless."
        )
    identity = identity or PackageIdentity()
    payload: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA_VERSION,
        "package": item.package,
        "facts": _facts_json(item, identity),
        "provenance": _provenance_json(item),
        "graph": _graph_json(graph, item.package),
        # The floor and every rule behind it, so the prompt can state the bound AND why it
        # exists. `RemovalFloor.as_json` already orders `fired` strictest-first.
        "floor": floor.as_json(),
        "anchors": [_anchor_json(entry) for entry in anchors],
    }
    return EvidenceBundle(package=item.package, payload=payload, sha256=bundle_sha256(payload))


def build_bundles(
    corpus: Sequence[CorpusPackage],
    *,
    floors: Mapping[str, RemovalFloor],
    identities: Mapping[str, PackageIdentity] | None = None,
    graph: CorpusGraph | None = None,
    upstream: UpstreamList | None = None,
    anchor_count: int = DEFAULT_ANCHOR_COUNT,
) -> dict[str, EvidenceBundle]:
    """A bundle for every package that has a floor, keyed by package name.

    Silently skipping a package with no floor would be the wrong shape — a package the ladder
    never saw is a pipeline-order problem, not a package to classify — so it raises.
    """
    identities = identities or {}
    bundles: dict[str, EvidenceBundle] = {}
    for item in sorted(corpus, key=lambda entry: entry.package):
        floor = floors.get(item.package)
        if floor is None:
            raise ValueError(
                f"build_bundles: {item.package} has no computed floor. The classification "
                "stage runs after rule_ladder, and a bundle with no lower bound cannot be "
                "validated against one; re-run the rule_ladder stage over this corpus."
            )
        anchors = (
            nearest_entries(item.package, upstream, limit=anchor_count)
            if upstream is not None and anchor_count > 0
            else ()
        )
        bundles[item.package] = build_bundle(
            item,
            floor=floor,
            identity=identities.get(item.package),
            graph=graph,
            anchors=anchors,
        )
    return bundles
