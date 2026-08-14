"""The corpus snapshot and the two mechanical dependency edges (`docs/pipeline-design.md` §4).

This is the stage that produces something no human contributor can produce at scale:
`uad_lists.json` has `dependencies` populated on 43 of 5372 entries and `neededBy` on 26, all
hand-curated. It is also the stage where a mistake is invisible — a wrong edge silently
changes a removal rating downstream — so **only two classes are emitted**, both of which a
reader can check against the manifest by hand:

- `<overlay android:targetPackage="X">` ⇒ the overlay depends on X. Fully mechanical: an RRO
  addresses exactly one target package and is meaningless without it.
- provider declares `<library name="L">` or `<static-library name="L">`, consumer declares
  `<uses-library name="L" required="true">` (`required` defaults to `"true"` per AOSP, so an
  attribute-less `<uses-library>` is the same consumer) or `<uses-static-library name="L">`
  ⇒ the consumer depends on the provider. AOSP documents the consumer as uninstallable on a
  device without the library.

Everything else is recorded as **evidence and never as an edge**, per the repo rule that
`dependencies` and `neededBy` are never inferred loosely:

- declared `<queries><package name="X">` — the caller is supposed to handle X being absent,
  so it is an interest, not a dependency;
- `<uses-library required="true">` names no corpus package provides. Measured on both local
  corpora, every one of these is a platform-provided Java shared library declared in
  `/etc/permissions/*.xml` (`android.test.base`, `com.google.android.dialer.support`, …), so
  the missing provider is the platform rather than a missing lookup;
- `content://<authority>` strings found in dex (`facts.parse_apk` scans them while the APK
  still exists), matched by authority string only — the record of the reference, never a
  resolution of which package it was aimed at. Split like the package queries: declared by
  another package in the corpus, or declared by nobody (off-corpus, or a runtime-built
  authority). Measured on the emulator corpus, the literal strings are mostly fragments
  (`%s` placeholders, off-corpus providers), so the absent half is the common case. The
  scan also tolerates one leading slash after the scheme — a join artifact of code-built
  URIs, unexercised by real data. The keys are present-and-empty on a package the scan
  found nothing on: an empty list now means "looked, found none", which is an honest claim.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Intent surfaces whose sole handler in the corpus is a rule-ladder input (design §6). MAIN
# plus the HOME category is how AOSP identifies a launcher; the other two are actions.
HOME_ACTION = "android.intent.action.MAIN"
HOME_CATEGORY = "android.intent.category.HOME"
DIALER_ACTION = "android.intent.action.DIAL"
SMS_DELIVER_ACTION = "android.provider.Telephony.SMS_DELIVER"

# The three edge/evidence kinds this module can produce.
EDGE_OVERLAY = "overlay"
EDGE_LIBRARY = "library"


@dataclass(frozen=True, slots=True)
class CorpusPackage:
    """One merged package as the deterministic core sees it.

    A plain value rather than the ORM row, for the same reason `facts.ApkFacts` is one: the
    graph, the ladder and the filter are pure functions over a corpus, they are exercised by
    fast tests over synthetic packages, and none of them may reach a database.
    """

    package: str
    devices: tuple[str, ...] = ()
    partitions: tuple[str, ...] = ()
    device_count: int = 1
    priv_app: bool = False
    core_app: bool = False
    persistent: bool = False
    shared_user_id: str | None = None
    overlay_target: str | None = None
    libraries: tuple[str, ...] = ()
    static_libraries: tuple[str, ...] = ()
    uses_libraries_required: tuple[str, ...] = ()
    # Carried but never read by `build_graph`: `required="false"` says the consumer is
    # designed to work without the library, so it is not a dependency. Present so that the
    # distinction is visible here rather than only in the parser.
    uses_libraries_optional: tuple[str, ...] = ()
    queries_packages: tuple[str, ...] = ()
    provider_authorities: tuple[str, ...] = ()
    # `content://` authorities referenced in this package's dex, matched by authority string
    # only. Evidence, never an edge: the reference may be optional, runtime-built, or aimed
    # off-corpus (`docs/pipeline-design.md` §4).
    content_uri_authorities: tuple[str, ...] = ()
    is_input_method: bool = False
    # (action, category) pairs flattened off the intent filters, so the ladder never has to
    # re-walk the filter structure to count core-intent handlers.
    handles_home: bool = False
    handles_dialer: bool = False
    handles_sms_deliver: bool = False

    @property
    def provided_libraries(self) -> tuple[str, ...]:
        """Every shared-library name this package declares, static or not. Both AOSP elements
        make the same promise to a `required="true"` consumer, so both are providers."""
        return tuple(sorted({*self.libraries, *self.static_libraries}))


@dataclass(frozen=True, slots=True)
class Edge:
    """One emitted dependency edge. `dependent` needs `provider` to exist."""

    kind: str
    dependent: str
    provider: str
    detail: str

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dependent": self.dependent,
            "provider": self.provider,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class PackageEvidence:
    """Relations recorded for the human and the model, never emitted as edges."""

    queries_packages_in_corpus: tuple[str, ...] = ()
    queries_packages_absent: tuple[str, ...] = ()
    # Required libraries no corpus package provides, split by whether `/etc/permissions`
    # shows the platform providing them.
    required_libraries_from_platform: tuple[str, ...] = ()
    required_libraries_unresolved: tuple[str, ...] = ()
    # Declared authorities another package in the corpus also declares. Zero on both local
    # corpora; a non-zero value is a genuine ambiguity a reviewer should see.
    shared_provider_authorities: tuple[str, ...] = ()
    # `content://` authorities referenced in the package's dex, split like the package
    # queries: declared by another package in the corpus (a real relation, matched by
    # authority string only — fuzzy by design), or declared by nobody (off-corpus or
    # runtime-selected). An authority the package declares itself appears in neither bucket.
    content_uri_authorities_in_corpus: tuple[str, ...] = ()
    content_uri_authorities_absent: tuple[str, ...] = ()
    # An `<overlay>` whose target is not in the corpus: the edge cannot be emitted, and the
    # reviewer should know the overlay is not orphaned, just aimed off-corpus.
    overlay_target_absent: str | None = None

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "queries_packages_in_corpus": list(self.queries_packages_in_corpus),
            "queries_packages_absent": list(self.queries_packages_absent),
            "required_libraries_from_platform": list(self.required_libraries_from_platform),
            "required_libraries_unresolved": list(self.required_libraries_unresolved),
            "shared_provider_authorities": list(self.shared_provider_authorities),
            "content_uri_authorities_in_corpus": list(self.content_uri_authorities_in_corpus),
            "content_uri_authorities_absent": list(self.content_uri_authorities_absent),
        }
        if self.overlay_target_absent is not None:
            payload["overlay_target_absent"] = self.overlay_target_absent
        return payload


@dataclass(frozen=True, slots=True)
class CorpusGraph:
    """Every emitted edge over one corpus, plus the evidence that was deliberately not one."""

    edges: tuple[Edge, ...] = ()
    evidence: Mapping[str, PackageEvidence] = field(default_factory=dict)

    def dependencies_of(self, package: str) -> tuple[str, ...]:
        return tuple(sorted({edge.provider for edge in self.edges if edge.dependent == package}))

    def needed_by(self, package: str) -> tuple[str, ...]:
        return tuple(sorted({edge.dependent for edge in self.edges if edge.provider == package}))

    def edges_for(self, package: str) -> tuple[Edge, ...]:
        return tuple(edge for edge in self.edges if package in (edge.dependent, edge.provider))

    def counts_by_kind(self) -> dict[str, int]:
        counts = {EDGE_OVERLAY: 0, EDGE_LIBRARY: 0}
        for edge in self.edges:
            counts[edge.kind] = counts.get(edge.kind, 0) + 1
        return counts


def library_providers(corpus: Iterable[CorpusPackage]) -> dict[str, tuple[str, ...]]:
    """Library name -> the corpus packages declaring it, sorted."""
    providers: dict[str, set[str]] = {}
    for item in corpus:
        for library in item.provided_libraries:
            providers.setdefault(library, set()).add(item.package)
    return {name: tuple(sorted(packages)) for name, packages in sorted(providers.items())}


def build_graph(
    corpus: Sequence[CorpusPackage], *, platform_libraries: frozenset[str] = frozenset()
) -> CorpusGraph:
    """Derive the two high-confidence edge classes and record the rest as evidence.

    Pure and order-independent: the same corpus in any order yields the same edges in the same
    order, because everything is sorted rather than emitted in iteration order.
    """
    names = {item.package for item in corpus}
    providers = library_providers(corpus)
    authority_owners: dict[str, set[str]] = {}
    for item in corpus:
        for authority in item.provider_authorities:
            authority_owners.setdefault(authority, set()).add(item.package)

    edges: set[Edge] = set()
    evidence: dict[str, PackageEvidence] = {}

    for item in corpus:
        overlay_absent: str | None = None
        if item.overlay_target:
            if item.overlay_target in names and item.overlay_target != item.package:
                edges.add(
                    Edge(
                        kind=EDGE_OVERLAY,
                        dependent=item.package,
                        provider=item.overlay_target,
                        detail=item.overlay_target,
                    )
                )
            else:
                overlay_absent = item.overlay_target

        from_platform: list[str] = []
        unresolved: list[str] = []
        for library in item.uses_libraries_required:
            owners = [owner for owner in providers.get(library, ()) if owner != item.package]
            if owners:
                for owner in owners:
                    edges.add(
                        Edge(
                            kind=EDGE_LIBRARY,
                            dependent=item.package,
                            provider=owner,
                            detail=library,
                        )
                    )
            elif library in platform_libraries:
                from_platform.append(library)
            else:
                unresolved.append(library)

        queried_present = sorted(name for name in item.queries_packages if name in names)
        queried_absent = sorted(name for name in item.queries_packages if name not in names)
        shared_authorities = sorted(
            authority
            for authority in item.provider_authorities
            if len(authority_owners.get(authority, ())) > 1
        )
        # An authority only the package itself declares is excluded from both buckets: a
        # package reaching its own provider is normal operation, not evidence about another
        # package — the same self-exclusion the library edge class applies.
        content_uri_in_corpus = sorted(
            authority
            for authority in item.content_uri_authorities
            if any(owner != item.package for owner in authority_owners.get(authority, ()))
        )
        content_uri_absent = sorted(
            authority
            for authority in item.content_uri_authorities
            if not authority_owners.get(authority)
        )
        evidence[item.package] = PackageEvidence(
            queries_packages_in_corpus=tuple(queried_present),
            queries_packages_absent=tuple(queried_absent),
            required_libraries_from_platform=tuple(sorted(from_platform)),
            required_libraries_unresolved=tuple(sorted(unresolved)),
            shared_provider_authorities=tuple(shared_authorities),
            content_uri_authorities_in_corpus=tuple(content_uri_in_corpus),
            content_uri_authorities_absent=tuple(content_uri_absent),
            overlay_target_absent=overlay_absent,
        )

    ordered = tuple(
        sorted(edges, key=lambda edge: (edge.kind, edge.dependent, edge.provider, edge.detail))
    )
    return CorpusGraph(edges=ordered, evidence=evidence)
